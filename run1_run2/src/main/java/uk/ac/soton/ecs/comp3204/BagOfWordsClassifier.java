package uk.ac.soton.ecs.comp3204;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class BagOfWordsClassifier implements Classifier {
    private LiblinearAnnotator<FImage, String> ann;

    @Override
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingImages) {
        // Building the vocabulary with the training data
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiliser(trainingImages);

        FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(assigner);

        // Constructing and training a linear classifier
        this.ann = new LiblinearAnnotator<>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(trainingImages);
    }

    @Override
    public String classify(FImage image) {
        ClassificationResult<String> result = this.ann.classify(image);
        return result.getPredictedClasses().iterator().next();
    }

    // Method for patch extraction
    static List<LocalFeature<SpatialLocation, FloatFV>> patchExtraction(FImage image){

        final List<LocalFeature<SpatialLocation, FloatFV>> patchList = new ArrayList<>();
        RectangleSampler rec = new RectangleSampler(image.normalise(), 4, 4, 8, 8);

        for(Rectangle rectangle : rec.allRectangles()){
            // Extracting a rectangular region of interest from an image
            FImage patch = image.extractROI(rectangle);

            // Reshape a 2D array into a 1D array
            final float[] vector = ArrayUtils.reshape(patch.pixels);
            final FloatFV featureV = new FloatFV(vector);

            // Assigning same location for rectangle and feature
            final SpatialLocation sl = new SpatialLocation(rectangle.x, rectangle.y);

            // Make the newly found feature local
            final LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<>(sl, featureV);

            // Add the feature to a list
            patchList.add(lf);
        }
        return patchList;
    }



    // Hard assigner from Chapter 12 from the OpenImaj Tutorial
    static HardAssigner<float[], float[], IntFloatPair> trainQuantiliser(Dataset<FImage> sample){
        List<float[]> allKeys = new ArrayList<>(); // patches flattened into vectors

        for(FImage img : sample){
            FImage image = img.getImage();
            // Getting a sample of random patches
            List<LocalFeature<SpatialLocation, FloatFV>> samplePatches = getRandomElements(patchExtraction(image), 15);
            // Putting the vector values of the feature vectors in a list
            for(LocalFeature<SpatialLocation, FloatFV> localFeature : samplePatches){
                allKeys.add(localFeature.getFeatureVector().values);
            }
        }

        // Performing K-Means clustering (creating 500 clusters)
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(700);
        float[][] dataSource = new float[allKeys.size()][];
        for(int i = 0; i < allKeys.size(); i++){
            dataSource[i] = allKeys.get(i);
        }
        FloatCentroidsResult result = km.cluster(dataSource);

        return result.defaultHardAssigner();
    }


    // Method to get random patches from the feature list
    static List<LocalFeature<SpatialLocation, FloatFV>> getRandomElements(List<LocalFeature<SpatialLocation, FloatFV>> patches, int numElements){
        List<LocalFeature<SpatialLocation, FloatFV>> randomList = new ArrayList<>();

        Random rand = new Random();

        for(int i = 0; i < numElements; i++){
            int index = rand.nextInt(patches.size());
            randomList.add(patches.get(index));
        }
        return randomList;
    }

    static class Extractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        public Extractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {

            // Bag of Visual Words uses the Hard Assigner to assign each feature to a visual word and compute the histograms
            final BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<>(assigner);
            final BlockSpatialAggregator<float[], SparseIntFV> blockSpatial = new BlockSpatialAggregator<>(bagOfVisualWords, 2, 2);
            final List<LocalFeature<SpatialLocation, FloatFV>> extractedFeature = patchExtraction(image);
            // Appending and normalising the spatial histograms
            return blockSpatial.aggregate(extractedFeature, image.getBounds()).normaliseFV();
        }
    }
}
