package dlchat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

@SuppressWarnings("serial")
public class LogsIterator implements MultiDataSetIterator {

    private static final boolean DEBUG = false;

    private List<List<Double>> logs;
    private int batchSize;
    private int totalBatches;
    private int currentBatch = 0;
    private int dictSize;
    private int rowSize;

    private Map<Double, String> revDict;

    public LogsIterator(List<List<Double>> logs, int batchSize, int dictSize, int rowSize, Map<Double, String> revDict) {
        this.logs = logs;
        this.batchSize = batchSize;
        this.dictSize = dictSize;
        this.rowSize = rowSize;
        this.revDict = revDict;
        totalBatches = logs.size() / batchSize + 1;
    }

    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public MultiDataSet next(int num) {
        long t1 = System.nanoTime();
        INDArray input = Nd4j.zeros(batchSize, 1, rowSize);
        INDArray prediction = Nd4j.zeros(batchSize, dictSize, rowSize);
        INDArray decode = Nd4j.zeros(batchSize, dictSize, rowSize);
        INDArray inputMask = Nd4j.zeros(batchSize, rowSize);
        INDArray predictionMask = Nd4j.zeros(batchSize, rowSize);
        long t2 = System.nanoTime();
//        System.out.println("Init time: " + (t2 - t1));
        int i = currentBatch * batchSize;
        for (int j = 0; j < batchSize; j++) {
            long t3 = System.nanoTime();
            if (i > logs.size() - 2) {
                break;
            }
            List<Double> rowIn = new ArrayList<>(logs.get(i));
            Collections.reverse(rowIn);
            List<Double> rowPred = new ArrayList<>(logs.get(i + 1));
            rowPred.add(1.0); // <eos>
            input.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.point(0), NDArrayIndex.interval(0, rowIn.size()) },
                    Nd4j.create(ArrayUtils.toPrimitive((Double[]) rowIn.toArray(new Double[0]))));
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

            long t4 = System.nanoTime();
//            System.out.println("Array fill time: " + (t4 - t3));
            if (DEBUG) {
                System.out.println("Row in: " + rowIn);
                INDArray inTensor = input.tensorAlongDimension(j, 1, 2);
                System.out.println("input tensor: " + inTensor);
                System.out.println("inputMask tensor: " + inputMask.tensorAlongDimension(j, 1));
                INDArray decodeTensor = decode.tensorAlongDimension(j, 1, 2);
                INDArray decodeMax = Nd4j.argMax(decodeTensor, 0);
                System.out.println("decodeMax tensor: " + decodeMax);
                System.out.println("decodeMask tensor: " + predictionMask.tensorAlongDimension(j, 1));
                INDArray predTensor = prediction.tensorAlongDimension(j, 1, 2);
                INDArray predMax = Nd4j.argMax(predTensor, 0);
                System.out.println("predMax tensor: " + predMax);
                System.out.println("predMask tensor: " + predictionMask.tensorAlongDimension(j, 1));
                System.out.print("IN: ");
                for (int sPos = 0; sPos < inTensor.size(1); sPos++) {
                    System.out.print(revDict.get(inTensor.getDouble(sPos)) + " ");
                }
                System.out.println();
                System.out.print("DECODE: ");
                for (int sPos = 0; sPos < decodeMax.size(1); sPos++) {
                    System.out.print(revDict.get(decodeMax.getDouble(sPos)) + " ");
                }
                System.out.println();
                System.out.print("OUT: ");
                for (int sPos = 0; sPos < predMax.size(1); sPos++) {
                    System.out.print(revDict.get(predMax.getDouble(sPos)) + " ");
                }
                System.out.println();
            }
            ++i;
            ++currentBatch;
        }
        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { input, decode }, new INDArray[] { prediction },
                new INDArray[] { inputMask, predictionMask }, new INDArray[] { predictionMask });
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        currentBatch = 0;
    }

    public int batch() {
        return currentBatch;
    }

    public int totalBatches() {
        return totalBatches;
    }

    public void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
    }

}
