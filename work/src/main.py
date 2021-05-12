import sys, os, shutil, math, random, time, subprocess, json
import time, datetime
import argparse, itertools

import vocabulary
from network import Network
from utils import out, format_elapsed
from const import START, STOP
from dataset import load_train_data, load_test_data, load_test_label_data

from collections import defaultdict

import paddle
import paddle.fluid as fluid
import numpy as np



def process_vocabulary(args, data, quiet=False):
    """
    Creates and returns vocabulary objects.
    Only iterates through the first 100 sequences, to save computation.
    """
    if not quiet:
        out(args.logfile, "initializing vacabularies... ", end="")
    seq_vocab = vocabulary.Vocabulary()
    bracket_vocab = vocabulary.Vocabulary()
    #loop_type_vocab = vocabulary.Vocabulary()

    for vocab in [seq_vocab, bracket_vocab]:#, loop_type_vocab]:
        vocab.index(START)
        vocab.index(STOP)
    for x in data[:100]:
        seq = x["sequence"]
        dot = x["structure"]
        #loop = x["predicted_loop_type"]
        for character in seq:
            seq_vocab.index(character)
        for character in dot:
            bracket_vocab.index(character)
        #for character in loop:
        #    loop_type_vocab.index(character)
    for vocab in [seq_vocab, bracket_vocab]:#, loop_type_vocab]:
        #vocab.index(UNK)
        vocab.freeze()
    if not quiet:
        out(args.logfile, "done.")
            
    def print_vocabulary(name, vocab):
        #special = {START, STOP, UNK}
        special = {START, STOP}
        out(args.logfile, "{}({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))
    if not quiet:
        print_vocabulary("Sequence", seq_vocab)
        print_vocabulary("Brackets", bracket_vocab)
    return seq_vocab, bracket_vocab
    

def reader_creator(args, data,
                   sequence_vocabulary, bracket_vocabulary,
                   test=False):
    def reader():
        for i,x in enumerate(data):
            seq = x["sequence"]
            dot = x["structure"]
            sequence = np.array([sequence_vocabulary.index(x) for x in list(seq)])
            structure = np.array([bracket_vocabulary.index(x) for x in list(dot)])
            if not test:
                LP_v_unpaired_prob = x["p_unpaired"]
                LP_v_unpaired_prob = np.array([x for x in LP_v_unpaired_prob])
                yield sequence, structure, LP_v_unpaired_prob
            else:
                yield sequence, structure
    return reader


def run_train(args):
    out(args.logfile, datetime.datetime.now())
    out(args.logfile, "# python3 " + " ".join(sys.argv))
    
    log = args.logfile
    train_data, val_data = load_train_data()
    out(log, "# Training set contains {} Sequences.".format(len(train_data)))
    out(log, "# Validation set contains {} Sequences.".format(len(val_data)))
            
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) if trainer_count > 1 else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    paddle.enable_static()
    out(log, "# Paddle: Using device: {}".format(place))
    out(log, "# Initializing model...")

    seq_vocab, bracket_vocab = process_vocabulary(args, train_data)    
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=args.dmodel,
        layers=args.layers,
        dropout=args.dropout,
    )
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    current_processed, total_processed = 0, 0
    check_every = math.floor((len(train_data) /  args.checks_per_epoch))
    best_dev_loss, best_dev_model_path = np.inf, None

    start_time = time.time()
    out(log, "# Checking validation {} times an epoch (every {} batches)".format(args.checks_per_epoch, check_every))
    patience = check_every * args.checks_per_epoch * 2
    batches_since_dev_update = 0

    train_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(args, train_data, seq_vocab, bracket_vocab), buf_size=500),
        batch_size=args.batch_size)
    val_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(args, val_data, seq_vocab, bracket_vocab), buf_size=500),
        batch_size=1)

    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    y = fluid.data(name="label", shape=[None], dtype="float32")
    predictions = network(seq, dot)
    
    loss = fluid.layers.mse_loss(input=predictions, label=y)
    avg_loss = fluid.layers.mean(loss)

    test_program = main_program.clone(for_test=True)
    feeder = paddle.fluid.DataFeeder(place=place, feed_list=[seq, dot, y])
    
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon=1e-08
    optimizer = fluid.optimizer.Adam(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    optimizer.minimize(avg_loss)
    exe.run(startup_program)
    exe_test = fluid.Executor(place)
    
    start_epoch_index = 1
    for epoch in itertools.count(start=start_epoch_index):
        if epoch >= args.epochs + 1:
            break
        train_reader = fluid.io.batch(
            fluid.io.shuffle(
                reader_creator(args, train_data, seq_vocab, bracket_vocab), buf_size=500),
        batch_size=args.batch_size)
        
        out(log, "# Epoch {} starting.".format(epoch))
        epoch_start_time = time.time()
        for batch_index, batch in enumerate(train_reader()):
            batch_loss,pred_values = exe.run(main_program, feed=feeder.feed(batch),
                                             fetch_list=[avg_loss.name, predictions.name],
                                             return_numpy=False)
            batch_loss = np.array(batch_loss)
            pred_values = np.array(pred_values)

            total_processed += len(batch)
            current_processed += len(batch)
            batches_since_dev_update += 1
            out(log,
                "epoch {:,} "
                "batch {:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {} "
                "".format(
                    epoch,
                    batch_index + 1,
                    total_processed,
                    float(batch_loss),
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )
            if math.isnan(float(batch_loss[0])):
                sys.exit("got NaN loss, training failed.")
            if current_processed >= check_every:
                current_processed -= (check_every)

                val_results = []
                for data in val_reader():
                    loss, pred = exe.run(test_program,
                                         feed=feeder.feed(data),
                                         fetch_list=[avg_loss.name, predictions.name],
                                         return_numpy=False
                    )
                    loss = np.array(loss)
                    val_results.append(loss[0])
                val_loss = sum(val_results) / len(val_results)
                out(log, "# Dev Average Loss: {:5.3f} (MSE) -> {:5.3f} (RMSD)".format(float(val_loss), math.sqrt(float(val_loss))))
                if val_loss < best_dev_loss:
                    batches_since_dev_update = 0
                    if best_dev_model_path is not None:
                        path = "{}/{}_dev={:.4f}".format(args.model_path_base, args.model_path_base, best_dev_loss)

                        print("\t\t",best_dev_model_path, os.path.exists(path))
                        if os.path.exists(path):
                            out(log, "* Removing previous model file {}...".format(path))
                            shutil.rmtree(path)
                    best_dev_loss = val_loss
                    best_dev_model_path = "{}_dev={:.4f}".format(args.model_path_base, val_loss)
                    out(log, "* Saving new best model to {}...".format(best_dev_model_path))
                    if not os.path.exists(args.model_path_base):
                        os.mkdir(args.model_path_base)
                    fluid.io.save_inference_model(args.model_path_base + "/" + best_dev_model_path, ['seq', 'dot'], [predictions], exe)




                
def run_test_withlabel(args):
    out(args.logfile, datetime.datetime.now())
    out(args.logfile, "# python3 " + " ".join(sys.argv))

    log = args.logfile
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id()) if trainer_count > 1 else fluid.CUDAPlace(0)
    out(log, "Loading data...")
    train_data, val_data = load_train_data()
    test_data = load_test_label_data()
    
    out(log, "Loading model...")
    seq_vocab, bracket_vocab = process_vocabulary(args, train_data)    
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=args.dmodel,
        layers=args.layers,
        dropout=0,
    )
    
    exe = fluid.Executor(place)
    paddle.enable_static()
    fluid.io.load_inference_model(args.model_path_base, exe)
    val_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(args, val_data, seq_vocab, bracket_vocab), buf_size=500),
        batch_size=args.batch_size)
    test_reader = fluid.io.batch(
            reader_creator(args, test_data, seq_vocab, bracket_vocab),
        batch_size=args.batch_size)

    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    y = fluid.data(name="label", shape=[None], dtype="float32")
    predictions = network(seq, dot)
    loss = fluid.layers.mse_loss(input=predictions, label=y)
    avg_loss = fluid.layers.mean(loss)
    
    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)
    feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot, y])
    
    val_results = []
    for data in val_reader():
        loss, pred = exe.run(test_program,
                             feed=feeder.feed(data),
                             fetch_list=[avg_loss.name, predictions.name],
                             return_numpy=False
        )
        loss = np.array(loss)
        val_results.append(loss[0])
    val_loss = sum(val_results) / len(val_results)
    out(log, "#  Dev Average Loss: {:6.4f} (MSE) -> {:6.4f} (RMSD)".format(float(val_loss), math.sqrt(float(val_loss))))
    
    test_results = []
    avg_losses = []
    for data in test_reader():
        loss, pred, gold = exe.run(test_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[avg_loss.name, predictions.name, y.name],
                                   return_numpy=False
        )
        loss = np.array(loss)
        test_results.append(loss[0])
        pred = list(np.array(pred))
        gold = list(np.array(gold))
        """
        print("PRED", ["{:5.3f}".format(x) for x in pred[:20]], "...")
        print("GOLD", ["{:5.3f}".format(x) for x in gold[:20]], "...")
        MSE = []
        for p,g in zip(pred, gold):
            mse = (p - g) ** 2
            MSE.append(mse)
        avg_mse = sum(MSE) / len(MSE)
        print("MSE ", ["{:5.3f}".format(x) for x in MSE[:20]], "...")
        print("AVG LOSS:", avg_mse)
        print()
        avg_losses.append(avg_mse)
        """
    test_loss = sum(test_results) / len(test_results)
    out(log, "# Test Average Loss: {:6.4f} (MSE) -> {:6.4f} (RMSD)".format(float(test_loss), math.sqrt(float(test_loss))))


def run_test(args):
    log = args.logfile
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id()) if trainer_count > 1 else fluid.CUDAPlace(0)
    print("Loading data...")
    train_data, val_data = load_train_data()
    test_data = load_test_data()

    print("Loading model...")
    seq_vocab, bracket_vocab = process_vocabulary(args, train_data, quiet=True)
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=args.dmodel,
        layers=args.layers,
        dropout=0,
    )
    
    exe = fluid.Executor(place)
    paddle.enable_static()
    fluid.io.load_inference_model(args.model_path_base, exe)
    test_reader = fluid.io.batch(
            reader_creator(args, test_data, seq_vocab, bracket_vocab, test=True),
        batch_size=args.batch_size)

    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    predictions = network(seq, dot)
    
    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)
    test_feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot])

    test_results = []
    for data in test_reader():
        pred, = exe.run(test_program,
                        feed=test_feeder.feed(data),
                        fetch_list=[predictions.name],
                        return_numpy=False
        )
        pred = list(np.array(pred))
        test_results.append(pred)
        out(log, " ".join([str(x) for x in pred]))

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--logfile", default="train_log.txt")
    subparser.add_argument("--batch-size", default=1)
    subparser.add_argument("--epochs", type=int, default=10)
    subparser.add_argument("--checks-per-epoch", type=float, default=10)
    subparser.add_argument("--dmodel", type=int, default=128)
    subparser.add_argument("--layers", type=int, default=8)
    subparser.add_argument("--dropout", type=float, default=0.15)
    
    subparser = subparsers.add_parser("test_withlabel")
    subparser.set_defaults(callback=run_test_withlabel)
    subparser.add_argument("--model-path-base", required=False)
    subparser.add_argument("--logfile", default="test_log.txt")
    subparser.add_argument("--batch-size", default=1)
    subparser.add_argument("--dmodel", type=int, default=128)
    subparser.add_argument("--layers", type=int, default=8)
    subparser.add_argument("--dropout", type=float, default=0.15)
    
    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=False)
    subparser.add_argument("--logfile", default="test_log.txt")
    subparser.add_argument("--batch-size", default=1)
    subparser.add_argument("--dmodel", type=int, default=128)
    subparser.add_argument("--layers", type=int, default=8)
    subparser.add_argument("--dropout", type=float, default=0.15)

    
    args = parser.parse_args()
    args.logfile = open(args.logfile, "w")
    args.callback(args)
    

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
