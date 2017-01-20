package dlchat;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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
    // RNN dimensions
    public static final int HIDDEN_LAYER_WIDTH = 1024;
    public static final int HIDDEN_LAYER_CONT = 1;
    private static final String FILENAME = "/home/rkfg/logs.txt";
    private static final int MINIBATCH_SIZE = 32;
    private static final int MAX_OUTPUT = 50;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(10);
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1);
    private static final int MAX_DICT = 2000;

    public static void main(String[] args) throws IOException {
        int idx = 2;
        dict.put("<unk>", 0);
        revDict.put(0, "<unk>");
        dict.put("<eol>", 1);
        revDict.put(1, "<eol>");
        for (char c : CHARS.toCharArray()) {
            if (!dict.containsKey(c)) {
                dict.put(String.valueOf(c), idx);
                revDict.put(idx, String.valueOf(c));
                ++idx;
            }
        }
        StatsStorage statsStorage = null;
        if (args.length == 0) {
            UIServer uiServer = UIServer.getInstance();
            statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
        }
        prepareData(idx);

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(3).learningRate(1e-1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                .miniBatch(true).updater(Updater.RMSPROP).weightInit(WeightInit.XAVIER);

        GraphBuilder graphBuilder = builder.graphBuilder();
        graphBuilder.addInputs("firstLine", "secondLine").setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("encoder",
                        new GravesLSTM.Builder().nIn(dict.size()).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.SOFTSIGN).build(),
                        "firstLine")
                .addVertex("lastTimeStep", new LastTimeStepVertex("firstLine"), "encoder")
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("secondLine"), "lastTimeStep")
                .addLayer("decoder",
                        new GravesLSTM.Builder().nIn(dict.size() + HIDDEN_LAYER_WIDTH).nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.SOFTSIGN).build(),
                        "secondLine", "duplicateTimeStep")
                .addLayer("output",
                        new RnnOutputLayer.Builder().nIn(HIDDEN_LAYER_WIDTH).nOut(dict.size()).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                        "decoder")
                .setOutputs("output").pretrain(false).backprop(true);

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
            net.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));
            learn(net, networkFile);
        }
    }

    private static void learn(ComputationGraph net, File networkFile) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        for (int epoch = 0; epoch < 100; ++epoch) {
            System.out.println("Epoch " + epoch);
            int i = 0;
            int lastPerc = 0;
            while (i <= logs.size() - 2) {
                int rowSize = 0;
                int batchSize = 0;
                for (int j = 0; j < MINIBATCH_SIZE + 1; j++) {
                    if (i + j >= logs.size()) {
                        break;
                    }
                    int curSize = logs.get(i + j).size();
                    if (curSize > rowSize) {
                        rowSize = curSize;
                    }
                    batchSize++;
                }
                INDArray input = Nd4j.zeros(batchSize, dict.size(), rowSize);
                INDArray decode = Nd4j.zeros(batchSize, dict.size(), rowSize);
                INDArray prediction = Nd4j.zeros(batchSize, dict.size(), rowSize);
                INDArray inputMask = Nd4j.zeros(batchSize, rowSize);
                INDArray decodeMask = Nd4j.zeros(batchSize, rowSize);
                INDArray predictionMask = Nd4j.zeros(batchSize, rowSize);
                for (int j = 0; j < MINIBATCH_SIZE; j++) {
                    if (i > logs.size() - 2) {
                        break;
                    }
                    List<Integer> rowIn = new ArrayList<>(logs.get(i));
                    Collections.reverse(rowIn);
                    List<Integer> rowPred = logs.get(i + 1);
                    for (int samplePos = 0; samplePos < rowSize; samplePos++) {
                        if (samplePos < rowIn.size()) {
                            input.putScalar(new int[] { j, rowIn.get(samplePos), samplePos }, 1);
                            inputMask.putScalar(new int[] { j, samplePos }, 1);
                        }
                        if (samplePos < rowPred.size()) {
                            prediction.putScalar(new int[] { j, rowPred.get(samplePos), samplePos }, 1);
                            predictionMask.putScalar(new int[] { j, samplePos }, 1);
                            decode.putScalar(new int[] { j, 0, samplePos }, 1);
                            decodeMask.putScalar(new int[] { j, samplePos }, 1);
                        }
                    }
                    i++;
                }
                net.fit(new INDArray[] { input, decode }, new INDArray[] { prediction }, new INDArray[] { inputMask, decodeMask },
                        new INDArray[] { predictionMask });
                int newPerc = (i * 100 / (logs.size() - 1));
                if (newPerc != lastPerc) {
                    System.out.println("Epoch complete: " + newPerc + "%");
                    lastPerc = newPerc;
                    if (newPerc % 5 == 0) {
                    }
                    // test(net);
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
            String line = "me|" + System.console().readLine() + "\n";
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
        ModelSerializer.writeModel(net, networkFile, true);
        System.out.println("Done.");
    }

    public static void test(ComputationGraph net) {
        System.out.println("======================== TEST ========================");
        List<Integer> rowIn = new ArrayList<>(logs.get(rnd.nextInt(logs.size())));
        System.out.println("In: ");
        for (Integer idx : rowIn) {
            System.out.print(revDict.get(idx) + " ");
        }
        System.out.println();
        System.out.print("Out: ");
        output(net, rowIn, true);
        System.out.println("======================== TEST END ========================");
    }

    private static void output(ComputationGraph net, List<Integer> rowIn, boolean printUnknowns) {
        Collections.reverse(rowIn);
        INDArray testIn = Nd4j.zeros(1, dict.size(), rowIn.size());
        int samplePos = 0;
        INDArray testDecode = Nd4j.zeros(1, dict.size(), MAX_OUTPUT);
        for (Integer currentChar : rowIn) {
            testIn.putScalar(new int[] { 0, currentChar, samplePos }, 1);
            if (samplePos < MAX_OUTPUT) {
                testDecode.putScalar(new int[] { 0, 0, samplePos }, 1);
            }
            samplePos++;
        }
        INDArray[] prediction_array = net.output(testIn, testDecode);
        INDArray predictions = prediction_array[0];
        INDArray answers = Nd4j.argMax(predictions, 1);
        for (int i = 0; i < answers.size(1); ++i) {
            int idx = answers.getInt(i);
            if (printUnknowns || !printUnknowns && idx != 0) {
                System.out.print(revDict.get(idx) + " ");
            }
            if (idx == 1) {
                break;
            }
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
                set = new HashSet<>();
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
