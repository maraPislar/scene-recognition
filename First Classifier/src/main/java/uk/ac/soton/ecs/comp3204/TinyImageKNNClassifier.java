package uk.ac.soton.ecs.comp3204;

import org.apache.lucene.analysis.util.CharArrayMap;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

import java.util.*;

public class TinyImageKNNClassifier {
    private final VFSGroupDataset<FImage> trainingImages;
    private DoubleNearestNeighboursExact kNearestNeighbours;
    private List<String> featureVectorGroups = new ArrayList<>();

    public TinyImageKNNClassifier(VFSGroupDataset<FImage> trainingImages) {
        this.trainingImages = trainingImages;
        this.train();
    }

    private void train() {
        List<double[]> featureVectors = new ArrayList<>();
        /*
        for(String group : this.trainingImages.getGroups()) {
            for (FImage trainingImage : this.trainingImages.get(group)) {
                double[] featureVector = getFeatureVector(trainingImage);
                featureVectors.add(featureVector);
                featureVectorGroups.add(group);
            }
        }
         */
        this.trainingImages.getGroups().forEach(group ->
            this.trainingImages.get(group).forEach(trainingImage -> {
                    featureVectors.add(getFeatureVector(trainingImage));
                    featureVectorGroups.add(group);
                }
            ));

        double[][] featureVectorsArray = new double[featureVectors.size()][featureVectors.get(0).length];
        featureVectors.toArray(featureVectorsArray);

        kNearestNeighbours = new DoubleNearestNeighboursExact(featureVectorsArray);
    }

    private static double[] getFeatureVector(FImage image) {
        int size = Math.min(image.width, image.height);
        FImage squareImage = image.extractCenter(size, size);
        FImage tinyImage = squareImage.process(new ResizeProcessor(16, 16));

        DoubleFV featureVector = new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(tinyImage.pixels)));
        featureVector.normaliseFV();

        return featureVector.values;
    }

    public String classify(FImage image, int k) {
        double[] imageFeatureVector = getFeatureVector(image);
        List<IntDoublePair> neighbours = kNearestNeighbours.searchKNN(imageFeatureVector, k);

        Map<String, Integer> neighboursGroupCount = new HashMap<>();

        for(IntDoublePair neighbour : neighbours) {
            String result = this.featureVectorGroups.get(neighbour.getFirst());
            int newGroupCount = neighboursGroupCount.getOrDefault(result, 0) + 1;
            neighboursGroupCount.put(result, newGroupCount);
        }

        Map.Entry<String, Integer> highestEntry = null;
        for(Map.Entry<String, Integer> groupCount : neighboursGroupCount.entrySet()) {
            if(highestEntry == null || groupCount.getValue() > highestEntry.getValue()) {
                highestEntry = groupCount;
            }
        }

        assert highestEntry != null;
        return highestEntry.getKey();
    }
}
