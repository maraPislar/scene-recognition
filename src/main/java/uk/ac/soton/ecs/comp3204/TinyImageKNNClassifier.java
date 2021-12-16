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
        //Iterate through every image of every group calculating the feature vector
        trainingImages.getGroups().forEach(group ->
            trainingImages.get(group).forEach(trainingImage -> {
                this.featureVectors.add(getFeatureVector(trainingImage));
                //Store corresponding group for reference later
                this.featureVectorGroups.add(group);
            })
        );

        //Convert from list to 2-d array
        double[][] featureVectorsArray = new double[this.featureVectors.size()][this.featureVectors.get(0).length];
        this.featureVectors.toArray(featureVectorsArray);

        //Initialise NearestNeigbours
        this.kNearestNeighbours = new DoubleNearestNeighboursExact(featureVectorsArray);
    }

    private static double[] getFeatureVector(FImage image) {
        //Calculate the smallest dimension
        int size = Math.min(image.width, image.height);
        //Extract square image of this size and resize
        FImage squareImage = image.extractCenter(size, size);
        FImage tinyImage = squareImage.process(new ResizeProcessor(16, 16));

        //Normalise feature vector
        DoubleFV featureVector = new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(tinyImage.pixels)));
        featureVector.normaliseFV();

        return featureVector.values;
    }

    @Override
    public String classify(FImage image) {
        //Convert image to classify into feature vector
        double[] imageFeatureVector = getFeatureVector(image);
        List<IntDoublePair> neighbours = this.kNearestNeighbours.searchKNN(imageFeatureVector, k);

        Map<String, Integer> neighboursGroupCount = new HashMap<>();

        for(IntDoublePair neighbour : neighbours) {
            //Get group corresponding to each resultant feature vector by index
            String result = this.featureVectorGroups.get(neighbour.getFirst());
            //Increment count
            int newGroupCount = neighboursGroupCount.getOrDefault(result, 0) + 1;
            neighboursGroupCount.put(result, newGroupCount);
        }

        //Calculate group with highest number of "votes"
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
