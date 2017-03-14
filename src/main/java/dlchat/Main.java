package dlchat;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.function.BiPredicate;
import java.util.function.Consumer;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {

    public static final Map<String, Integer> dict = new HashMap<>();
    public static final Map<Integer, String> revDict = new HashMap<>();
    private static final String CHARS = "-\\/_&" + LogProcessor.SPECIALS;
    private static List<List<Integer>> logs = new ArrayList<>();
    private static Random rng = new Random();
    // RNN dimensions
    public static final int HIDDEN_LAYER_WIDTH = 2048;
    public static final int HIDDEN_LAYER_CONT = 1;
    private static final String FILENAME = "/home/rkfg/movie_lines_trunc.txt";
    private static final String BACKUP_FILENAME = "/home/rkfg/rnn_train.bak.zip";
    private static final int MINIBATCH_SIZE = 128;
    private static final int MAX_OUTPUT = 50;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(20);
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1);
    private static final int MAX_DICT = 10000;
    private static final int TBPTT_SIZE = 50;
    private static final double LEARNING_RATE = 1e-4;
    private static final double L2 = 1e-3;
    private static final double RMS_DECAY = 0.95;
    private static final int ROW_SIZE = 100;
    private static final boolean DEBUG = false;

    public static void main(String[] args) throws IOException {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        cleanupTmp();
        int idx = 2;
        dict.put("<unk>", 0);
        revDict.put(0, "<unk>");
        dict.put("<eos>", 1);
        revDict.put(1, "<eos>");
        for (char c : CHARS.toCharArray()) {
            if (!dict.containsKey(c)) {
                dict.put(String.valueOf(c), idx);
                revDict.put(idx, String.valueOf(c));
                ++idx;
            }
        }
        prepareData(idx);

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(1).learningRate(LEARNING_RATE).rmsDecay(RMS_DECAY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123).miniBatch(true).updater(Updater.RMSPROP)
                .weightInit(WeightInit.XAVIER).regularization(true).l2(L2)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        GraphBuilder graphBuilder = builder.graphBuilder().pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                .tBPTTBackwardLength(TBPTT_SIZE).tBPTTForwardLength(TBPTT_SIZE);
        graphBuilder.addInputs("firstLine", "decoderDummy")
                .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("encoder", new GravesLSTM.Builder().nIn(dict.size()).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.TANH).build(),
                        "firstLine")
                .addVertex("thoughtVector", new LastTimeStepVertex("firstLine"), "encoder")
                .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderDummy"), "thoughtVector")
                .addLayer("decoder",
                        new GravesLSTM.Builder().nIn(dict.size() + HIDDEN_LAYER_WIDTH).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.TANH)
                                .build(),
                        "decoderDummy", "dup")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(HIDDEN_LAYER_WIDTH).nOut(dict.size()).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
                .setOutputs("output");

        ComputationGraphConfiguration conf = graphBuilder.build();
        ComputationGraph net;
        File networkFile = new File("/home/rkfg/rnn_train.zip");
        if (networkFile.exists()) {
            System.out.println("Loading the existing network...");
            net = ModelSerializer.restoreComputationGraph(networkFile);
            if (args.length == 0) {
                test(net);
            }
        } else {
            System.out.println("Creating a new network...");
            net = new ComputationGraph(conf);
            net.init();
        }

        if (args.length == 1 && args[0].equals("dialog")) {
            startDialog(net);
        } else {
            net.setListeners(new ScoreIterationListener(1));
            learn(net, networkFile);
        }
    }

    private static void learn(ComputationGraph net, File networkFile) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        INDArray input = Nd4j.zeros(MINIBATCH_SIZE, dict.size(), ROW_SIZE);
        INDArray prediction = Nd4j.zeros(MINIBATCH_SIZE, dict.size(), ROW_SIZE);
        INDArray decode = Nd4j.zeros(MINIBATCH_SIZE, dict.size(), ROW_SIZE);
        INDArray inputMask = Nd4j.zeros(MINIBATCH_SIZE, ROW_SIZE);
        INDArray predictionMask = Nd4j.zeros(MINIBATCH_SIZE, ROW_SIZE);
        INDArray decodeMask = Nd4j.zeros(MINIBATCH_SIZE, ROW_SIZE);
        for (int epoch = 0; epoch < 20; ++epoch) {
            System.out.println("Epoch " + epoch);
            // Collections.shuffle(logs);
            int i = 0;
            String shift = System.getProperty("dlchat.shift");
            if (epoch == 0 && shift != null) {
                i = Integer.valueOf(shift);
            }
            int lastPerc = 0;
            while (i < logs.size() - 1) {
                int prevI = i;
                for (int j = 0; j < MINIBATCH_SIZE; j++) {
                    if (i > logs.size() - 2) {
                        break;
                    }
                    List<Integer> rowIn = new ArrayList<>(logs.get(i));
                    Collections.reverse(rowIn);
                    List<Integer> rowPred = logs.get(i + 1);
                    for (int seq = 0; seq < ROW_SIZE; seq++) {
                        if (seq < rowIn.size()) {
                            int in = rowIn.get(seq);
                            input.putScalar(new int[] { j, in, seq }, 1);
                            inputMask.putScalar(new int[] { j, seq }, 1);
                        } else {
                            inputMask.putScalar(new int[] { j, seq }, 0);
                        }
                        if (seq < rowPred.size()) {
                            int pred = rowPred.get(seq);
                            prediction.putScalar(new int[] { j, pred, seq }, 1);
                            predictionMask.putScalar(new int[] { j, seq }, 1);
                            decodeMask.putScalar(new int[] { j, seq }, 1);
                        } else {
                            predictionMask.putScalar(new int[] { j, seq }, 0);
                            decodeMask.putScalar(new int[] { j, seq }, 0);
                        }
                    }
                    if (DEBUG) {
                        System.out.println("Row in: " + rowIn);
                        INDArray inTensor = input.tensorAlongDimension(j, 1, 2);
                        INDArray inputMax = Nd4j.argMax(inTensor, 0);
                        System.out.println("inputMax tensor: " + inputMax);
                        System.out.println("inputMask tensor: " + inputMask.tensorAlongDimension(j, 1));
                        INDArray predTensor = prediction.tensorAlongDimension(j, 1, 2);
                        INDArray predMax = Nd4j.argMax(predTensor, 0);
                        System.out.println("predMax tensor: " + predMax);
                        System.out.println("predMask tensor: " + predictionMask.tensorAlongDimension(j, 1));
                        System.out.println("decodeMask tensor: " + decodeMask.tensorAlongDimension(j, 1));
                        System.out.print("IN: ");
                        for (int sPos = 0; sPos < inputMax.size(1); sPos++) {
                            System.out.print(revDict.get(inputMax.getInt(sPos)) + " ");
                        }
                        System.out.println();
                        System.out.print("OUT: ");
                        for (int sPos = 0; sPos < predMax.size(1); sPos++) {
                            System.out.print(revDict.get(predMax.getInt(sPos)) + " ");
                        }
                        System.out.println();
                    }
                    ++i;
                }
                net.fit(new INDArray[] { input, decode }, new INDArray[] { prediction }, new INDArray[] { inputMask, decodeMask },
                        new INDArray[] { predictionMask });
                if (net.score() < 0) {
                    if (DEBUG) {
                        for (int j = 0; j < MINIBATCH_SIZE; ++j) {
                            INDArray inputMax = Nd4j.argMax(input.tensorAlongDimension(j, 1, 2), 0);
                            System.out.println("inputMax tensor: " + inputMax);
                            System.out.println("inputMask tensor: " + inputMask.tensorAlongDimension(j, 1));
                            INDArray predMax = Nd4j.argMax(prediction.tensorAlongDimension(j, 1, 2), 0);
                            System.out.println("predMax tensor: " + predMax);
                            System.out.println("predMask tensor: " + predictionMask.tensorAlongDimension(j, 1));
                            System.out.print("IN: ");
                            for (int sPos = 0; sPos < inputMax.size(1); sPos++) {
                                System.out.print(revDict.get(inputMax.getInt(sPos)) + " ");
                            }
                            System.out.println();
                            System.out.print("OUT: ");
                            for (int sPos = 0; sPos < predMax.size(1); sPos++) {
                                System.out.print(revDict.get(predMax.getInt(sPos)) + " ");
                            }
                            System.out.println();
                        }
                    }
                }
                // reset everything
                for (int j = 0; j < MINIBATCH_SIZE; j++) {
                    if (prevI > logs.size() - 2) {
                        break;
                    }
                    List<Integer> rowIn = logs.get(prevI);
                    List<Integer> rowPred = logs.get(prevI + 1);
                    for (int seq = 0; seq < ROW_SIZE; seq++) {
                        if (seq < rowIn.size()) {
                            int in = rowIn.get(seq);
                            input.putScalar(new int[] { j, in, seq }, 0);
                            inputMask.putScalar(new int[] { j, seq }, 0);
                        }
                        if (seq < rowPred.size()) {
                            int pred = rowPred.get(seq);
                            prediction.putScalar(new int[] { j, pred, seq }, 0);
                            predictionMask.putScalar(new int[] { j, seq }, 0);
                            decodeMask.putScalar(new int[] { j, seq }, 0);
                        }
                    }
                    ++prevI;
                }
                System.out.println("I = " + i);
                int newPerc = (i * 100 / (logs.size() - 1));
                if (newPerc != lastPerc) {
                    System.out.println("Epoch complete: " + newPerc + "%");
                    lastPerc = newPerc;
                }
                if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
                    saveModel(net, networkFile);
                    lastSaveTime = System.currentTimeMillis();
                }
                if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
                    test(net);
                    lastTestTime = System.currentTimeMillis();
                }
            }
        }
    }

    private static void startDialog(ComputationGraph net) throws IOException {
        System.out.println("Dialog started.");
        while (true) {
            System.out.print("In> ");
            String line = "1 +++$+++ u11 +++$+++ m0 +++$+++ WALTER +++$+++ " + System.console().readLine() + "\n";
            LogProcessor dialogProcessor = new LogProcessor(new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), false) {
                @Override
                protected void processLine(String lastLine) {
                    List<String> words = new ArrayList<>();
                    doProcessLine(lastLine, words, true);
                    List<Integer> wordIdxs = new ArrayList<>();
                    if (processWords(words, wordIdxs)) {
                        System.out.print("Out> ");
                        output(net, wordIdxs, true);
                    }
                }
            };
            dialogProcessor.setDict(dict);
            dialogProcessor.start();
        }
    }

    private static void saveModel(ComputationGraph net, File networkFile) throws IOException {
        System.out.println("Saving the model...");
        File backup = new File(BACKUP_FILENAME);
        if (networkFile.exists()) {
            if (backup.exists()) {
                backup.delete();
            }
            networkFile.renameTo(backup);
        }
        ModelSerializer.writeModel(net, networkFile, true);
        cleanupTmp();
        System.out.println("Done.");
    }

    public static void cleanupTmp() throws IOException {
        Files.find(Paths.get("/tmp"), 1, new BiPredicate<Path, BasicFileAttributes>() {

            @Override
            public boolean test(Path t, BasicFileAttributes u) {
                return t.getFileName().toString().startsWith("model");
            }
        }).forEach(new Consumer<Path>() {

            @Override
            public void accept(Path t) {
                try {
                    Files.delete(t);
                } catch (IOException e) {
                    System.out.println("Can't delete " + t);
                    e.printStackTrace();
                }
            }
        });
    }

    public static void test(ComputationGraph net) {
        System.out.println("======================== TEST ========================");
        int selected = rnd.nextInt(logs.size());
        List<Integer> rowIn = new ArrayList<>(logs.get(selected));
        System.out.print("In: ");
        for (Integer idx : rowIn) {
            System.out.print(revDict.get(idx) + " ");
        }
        System.out.println();
        System.out.print("Out: ");
        output(net, rowIn, true);
        System.out.println("======================== TEST END ========================");
    }

    private static void output(ComputationGraph net, List<Integer> rowIn, boolean printUnknowns) {
        net.rnnClearPreviousState();
        Collections.reverse(rowIn);
        int size = Math.min(rowIn.size(), ROW_SIZE);
        INDArray in = Nd4j.zeros(1, dict.size(), size);
        INDArray decodeDummy = Nd4j.zeros(1, dict.size(), ROW_SIZE);
        for (int i = 0; i < size; ++i) {
            in.putScalar(0, rowIn.get(i), i, 1);
        }
        INDArray out = net.outputSingle(in, decodeDummy);
        // System.out.println("OUT SHAPE: " + out.shapeInfoToString());
        for (int i = 0; i < out.size(2); ++i) {
            double d = rng.nextDouble();
            double sum = 0.0;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, i);
                if (d <= sum) {
                    if (printUnknowns || !printUnknowns && s != 0) {
                        System.out.print(revDict.get(s) + " ");
                    }
                    break;
                }
            }
            // if (idx == 1) {
            // break;
            // }
        }
        System.out.println();
    }

    public static void prepareData(int idx) throws IOException, FileNotFoundException {
        System.out.println("Building the dictionary...");
        LogProcessor logProcessor = new LogProcessor(FILENAME, true);
        logProcessor.start();
        Map<String, Integer> freqs = logProcessor.getFreq();
        Set<String> dictSet = new TreeSet<>();
        Map<Integer, Set<String>> freqMap = new TreeMap<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (Entry<String, Integer> entry : freqs.entrySet()) {
            Set<String> set = freqMap.get(entry.getValue());
            if (set == null) {
                set = new TreeSet<>();
                freqMap.put(entry.getValue(), set);
            }
            set.add(entry.getKey());
        }
        int cnt = 0;
        dictSet.addAll(dict.keySet());
        for (Entry<Integer, Set<String>> entry : freqMap.entrySet()) {
            for (String val : entry.getValue()) {
                if (dictSet.add(val) && ++cnt >= MAX_DICT) {
                    break;
                }
            }
            if (cnt >= MAX_DICT) {
                break;
            }
        }
        System.out.println("Dictionary is ready, size is " + dictSet.size());
        for (String word : dictSet) {
            if (!dict.containsKey(word)) {
                dict.put(word, idx);
                revDict.put(idx, word);
                ++idx;
            }
        }
        System.out.println("Total dictionary size is " + dict.size() + ". Processing the dataset...");
        // System.out.println(dict);
        logProcessor = new LogProcessor(FILENAME, false) {
            @Override
            protected void processLine(String lastLine) {
                List<Integer> wordIdxs = new ArrayList<>();
                ArrayList<String> words = new ArrayList<>();
                doProcessLine(lastLine, words, true);
                if (!words.isEmpty()) {
                    if (processWords(words, wordIdxs)) {
                        logs.add(wordIdxs);
                    }
                }
            }
        };
        logProcessor.setDict(dict);
        logProcessor.start();
        System.out.println("Done. Logs size is " + logs.size());
    }

}
