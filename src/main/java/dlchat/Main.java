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
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.function.BiPredicate;
import java.util.function.Consumer;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
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
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {

    public static final Map<String, Double> dict = new HashMap<>();
    public static final Map<Double, String> revDict = new HashMap<>();
    private static final String CHARS = "-\\/_&" + LogProcessor.SPECIALS;
    private static List<List<Double>> logs = new ArrayList<>();
    private static Random rng = new Random();
    // RNN dimensions
    public static final int HIDDEN_LAYER_WIDTH = 512;
    private static final int EMBEDDING_WIDTH = 256;
    private static final String CORPUS_FILENAME = "movie_lines.txt";
    private static final String NETWORK_FILE_PATH = "rnn_train.zip";
    private static final String BACKUP_FILENAME = "rnn_train.bak.zip";
    private static final int MINIBATCH_SIZE = 64;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5);
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1);
    private static final int MAX_DICT = 10000;
    private static final int TBPTT_SIZE = 25;
    private static final double LEARNING_RATE = 1e-1;
    private static final double L2 = 1e-3;
    private static final double RMS_DECAY = 0.95;
    private static final int ROW_SIZE = 40;
    private static final int GC_WINDOW = 5000;
    private static final int MACROBATCH_SIZE = 10;

    public static void main(String[] args) throws IOException {
        new Main().run(args);
    }

    private void run(String[] args) throws IOException {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);
        Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
        configuration.setMaximumBlockSize(768);
        configuration.setMinimumBlockSize(512);

        // configuration.enableDebug(true);
        // configuration.setVerbose(true);

        cleanupTmp();
        double idx = 3.0;
        dict.put("<unk>", 0.0);
        revDict.put(0.0, "<unk>");
        dict.put("<eos>", 1.0);
        revDict.put(1.0, "<eos>");
        dict.put("<go>", 2.0);
        revDict.put(2.0, "<go>");
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
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).miniBatch(true).updater(Updater.RMSPROP)
                .weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        GraphBuilder graphBuilder = builder.graphBuilder().pretrain(false).backprop(true).backpropType(BackpropType.Standard)
                .tBPTTBackwardLength(TBPTT_SIZE).tBPTTForwardLength(TBPTT_SIZE);
        graphBuilder.addInputs("inputLine", "decoderInput")
                .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("embeddingEncoder", new EmbeddingLayer.Builder().nIn(dict.size()).nOut(EMBEDDING_WIDTH).build(), "inputLine")
                .addLayer("encoder",
                        new GravesLSTM.Builder().nIn(EMBEDDING_WIDTH).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.TANH).build(),
                        "embeddingEncoder")
                .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
                .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
                .addLayer("decoder",
                        new GravesLSTM.Builder().nIn(dict.size() + HIDDEN_LAYER_WIDTH).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.TANH)
                                .build(),
                        "decoderInput", "dup")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(HIDDEN_LAYER_WIDTH).nOut(dict.size()).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
                .setOutputs("output");

        ComputationGraphConfiguration conf = graphBuilder.build();
        ComputationGraph net;
        File networkFile = new File(NETWORK_FILE_PATH);
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

    private void learn(ComputationGraph net, File networkFile) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        LogsIterator logsIterator = new LogsIterator(logs, MINIBATCH_SIZE, MACROBATCH_SIZE, dict.size(), ROW_SIZE, revDict);
        for (int epoch = 1; epoch < 10000; ++epoch) {
            System.out.println("Epoch " + epoch);
            int i = 0;
            String shift = System.getProperty("dlchat.shift");
            if (epoch == 1 && shift != null) {
                logsIterator.setCurrentBatch(Integer.valueOf(shift));
            }
            int lastPerc = 0;
            logsIterator.reset();
            while (logsIterator.hasNextMacrobatch()) {
                long t1 = System.nanoTime();
                net.fit(logsIterator);
                long t2 = System.nanoTime();
                logsIterator.nextMacroBatch();
                System.out.println("Fit time: " + (t2 - t1));
                System.out.println("Batch = " + logsIterator.batch());
                int newPerc = (logsIterator.batch() * 100 / logsIterator.totalBatches());
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

    private void startDialog(ComputationGraph net) throws IOException {
        System.out.println("Dialog started.");
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("In> ");
                String line = "1 +++$+++ u11 +++$+++ m0 +++$+++ WALTER +++$+++ " + scanner.nextLine() + "\n";
                LogProcessor dialogProcessor = new LogProcessor(new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), ROW_SIZE,
                        false) {
                    @Override
                    protected void processLine(String lastLine) {
                        List<String> words = new ArrayList<>();
                        doProcessLine(lastLine, words, true);
                        List<Double> wordIdxs = new ArrayList<>();
                        if (processWords(words, wordIdxs)) {
                            System.out.print("Got words: ");
                            for (Double idx : wordIdxs) {
                                System.out.print(revDict.get(idx) + " ");
                            }
                            System.out.println();
                            System.out.print("Out> ");
                            output(net, wordIdxs, true, true);
                        }
                    }
                };
                dialogProcessor.setDict(dict);
                dialogProcessor.start();
            }
        }
    }

    private void saveModel(ComputationGraph net, File networkFile) throws IOException {
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

    private void cleanupTmp() throws IOException {
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

    private void test(ComputationGraph net) {
        System.out.println("======================== TEST ========================");
        int selected = rnd.nextInt(logs.size());
        List<Double> rowIn = new ArrayList<>(logs.get(selected));
        System.out.print("In: ");
        for (Double idx : rowIn) {
            System.out.print(revDict.get(idx) + " ");
        }
        System.out.println();
        System.out.print("Out: ");
        output(net, rowIn, true, true);
        System.out.println("======================== TEST END ========================");
    }

    private void output(ComputationGraph net, List<Double> rowIn, boolean printUnknowns, boolean stopOnEos) {
        net.rnnClearPreviousState();
        Collections.reverse(rowIn);
        INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
        double[] decodeArr = new double[dict.size()];
        decodeArr[2] = 1;
        INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dict.size(), 1 });
        net.outputSingle(in, decode);
        org.deeplearning4j.nn.layers.recurrent.GravesLSTM decoder = (org.deeplearning4j.nn.layers.recurrent.GravesLSTM) net
                .getLayer("decoder");
        Layer output = net.getLayer("output");
        GraphVertex mergeVertex = net.getVertex("decoder-merge");
        INDArray thoughtVector = mergeVertex.getInputs()[1];
        for (int row = 0; row < ROW_SIZE; ++row) {
            mergeVertex.setInputs(decode, thoughtVector);
            INDArray merged = mergeVertex.doForward(false);
            INDArray activateDec = decoder.rnnTimeStep(merged);
            INDArray out = output.activate(activateDec, false);
            double d = rng.nextDouble();
            double sum = 0.0;
            int idx = -1;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, 0);
                if (d <= sum) {
                    idx = s;
                    if (printUnknowns || s != 0) {
                        System.out.print(revDict.get((double) s) + " ");
                    }
                    break;
                }
            }
            if (stopOnEos && idx == 1) {
                break;
            }
            double[] newDecodeArr = new double[dict.size()];
            newDecodeArr[idx] = 1;
            decode = Nd4j.create(newDecodeArr, new int[] { 1, dict.size(), 1 });
        }
        System.out.println();
    }

    private void prepareData(double idx) throws IOException, FileNotFoundException {
        System.out.println("Building the dictionary...");
        LogProcessor logProcessor = new LogProcessor(CORPUS_FILENAME, ROW_SIZE, true);
        logProcessor.start();
        Map<String, Double> freqs = logProcessor.getFreq();
        Set<String> dictSet = new TreeSet<>();
        Map<Double, Set<String>> freqMap = new TreeMap<>(new Comparator<Double>() {

            @Override
            public int compare(Double o1, Double o2) {
                return (int) (o2 - o1);
            }
        });
        for (Entry<String, Double> entry : freqs.entrySet()) {
            Set<String> set = freqMap.get(entry.getValue());
            if (set == null) {
                set = new TreeSet<>();
                freqMap.put(entry.getValue(), set);
            }
            set.add(entry.getKey());
        }
        int cnt = 0;
        dictSet.addAll(dict.keySet());
        for (Entry<Double, Set<String>> entry : freqMap.entrySet()) {
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
        logProcessor = new LogProcessor(CORPUS_FILENAME, ROW_SIZE, false) {
            @Override
            protected void processLine(String lastLine) {
                List<Double> wordIdxs = new ArrayList<>();
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
