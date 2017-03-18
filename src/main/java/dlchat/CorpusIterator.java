package dlchat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

@SuppressWarnings("serial")
public class CorpusIterator implements MultiDataSetIterator {

    /*
     * Motivation: I want to get asynchronous data iteration while not blocking on net.fit() until the end of epoch. I want to checkpoint
     * the network, show intermediate test results and some stats, it would be harder to achieve with listeners I think so this is how I
     * solved the problem. This way the learn process is asynchronous inside one macrobatch and synchronous across all the macrobatches.
     * 
     * Macrobatch is a group of minibatches. The iterator is modified so that it reports the end of data when it exhausts a macrobatch. Then
     * it advances (manually) to the next macrobatch.
     */

    private List<List<Double>> logs;
    private int batchSize;
    private int batchesPerMacrobatch;
    private int totalBatches;
    private int totalMacroBatches;
    private int currentBatch = 0;
    private int currentMacroBatch = 0;
    private int dictSize;
    private int rowSize;

    public CorpusIterator(List<List<Double>> logs, int batchSize, int batchesPerMacrobatch, int dictSize, int rowSize) {
        this.logs = logs;
        this.batchSize = batchSize;
        this.batchesPerMacrobatch = batchesPerMacrobatch;
        this.dictSize = dictSize;
        this.rowSize = rowSize;
        totalBatches = logs.size() / batchSize + 1;
        totalMacroBatches = totalBatches / batchesPerMacrobatch + 1;
    }

    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches && getMacroBatchByCurrentBatch() == currentMacroBatch;
    }

    private int getMacroBatchByCurrentBatch() {
        return currentBatch / batchesPerMacrobatch;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public MultiDataSet next(int num) {
        INDArray input = Nd4j.zeros(batchSize, 1, rowSize);
        INDArray prediction = Nd4j.zeros(batchSize, dictSize, rowSize);
        INDArray decode = Nd4j.zeros(batchSize, dictSize, rowSize);
        INDArray inputMask = Nd4j.zeros(batchSize, rowSize);
        INDArray predictionMask = Nd4j.zeros(batchSize, rowSize);
        int i = currentBatch * batchSize;
        for (int j = 0; j < batchSize; j++) {
            if (i > logs.size() - 2) {
                break;
            }
            List<Double> rowIn = new ArrayList<>(logs.get(i));
            Collections.reverse(rowIn);
            List<Double> rowPred = new ArrayList<>(logs.get(i + 1));
            rowPred.add(1.0); // <eos>
            input.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.point(0), NDArrayIndex.interval(0, rowIn.size()) },
                    Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0]))));
            inputMask.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, rowIn.size()) }, Nd4j.ones(rowIn.size()));
            predictionMask.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, rowPred.size()) },
                    Nd4j.ones(rowPred.size()));
            double predOneHot[][] = new double[dictSize][rowPred.size()];
            double decodeOneHot[][] = new double[dictSize][rowPred.size()];
            decodeOneHot[2][0] = 1;
            int predIdx = 0;
            for (Double pred : rowPred) {
                predOneHot[pred.intValue()][predIdx] = 1;
                if (predIdx < rowPred.size() - 1) {
                    decodeOneHot[pred.intValue()][predIdx + 1] = 1;
                }
                ++predIdx;
            }
            prediction.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size()) }, Nd4j.create(predOneHot));
            decode.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size()) }, Nd4j.create(decodeOneHot));
            ++i;
        }
        ++currentBatch;
        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { input, decode }, new INDArray[] { prediction },
                new INDArray[] { inputMask, predictionMask }, new INDArray[] { predictionMask });
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        currentBatch = 0;
        currentMacroBatch = 0;
    }

    public int batch() {
        return currentBatch;
    }

    public int totalBatches() {
        return totalBatches;
    }

    public void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
        currentMacroBatch = getMacroBatchByCurrentBatch();
    }

    public boolean hasNextMacrobatch() {
        return getMacroBatchByCurrentBatch() < totalMacroBatches && currentMacroBatch < totalMacroBatches;
    }

    public void nextMacroBatch() {
        ++currentMacroBatch;
    }

}
