package uk.ac.soton.ecs.comp3204;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

import java.util.*;

public class TinyImageKNNClassifier implements Classifier {
    private final int k;
    private List<double[]> featureVectors = new ArrayList<>();
    private List<String> featureVectorGroups = new ArrayList<>();
    private DoubleNearestNeighboursExact kNearestNeighbours;

    public TinyImageKNNClassifier(int k) {
        this.k = k;
    }

    @Override
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingImages) {
        trainingImages.getGroups().forEach(group ->
            trainingImages.get(group).forEach(trainingImage -> {
                this.featureVectors.add(getFeatureVector(trainingImage));
                this.featureVectorGroups.add(group);
            })
        );

        double[][] featureVectorsArray = new double[this.featureVectors.size()][this.featureVectors.get(0).length];
        this.featureVectors.toArray(featureVectorsArray);

        this.kNearestNeighbours = new DoubleNearestNeighboursExact(featureVectorsArray);
    }

    private static double[] getFeatureVector(FImage image) {
        int size = Math.min(image.width, image.height);
        FImage squareImage = image.extractCenter(size, size);
        FImage tinyImage = squareImage.process(new ResizeProcessor(16, 16));

        DoubleFV featureVector = new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(tinyImage.pixels)));
        featureVector.normaliseFV();

        return featureVector.values;
    }

    @Override
    public String classify(FImage image) {
        double[] imageFeatureVector = getFeatureVector(image);
        List<IntDoublePair> neighbours = this.kNearestNeighbours.searchKNN(imageFeatureVector, k);

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
