package uk.ac.soton.ac.uk.3204;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class App {
    public static void main( String[] args ) throws FileSystemException {
        //Loading the dataset
        VFSGroupDataset<FImage> images = new VFSGroupDataset<>("D:\\Uni\\Year 3\\Vision\\coursework3\\training", ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> testing = new VFSGroupDataset<>("D:\\Uni\\Year 3\\Vision\\coursework3\\testing", ImageUtilities.FIMAGE_READER);

        // Building the vocabulary with the training data
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiliser(images);

        BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<>(assigner);
        
    }


    // Method for patch extraction
    static List<LocalFeature<SpatialLocation, FloatFV>> patchExtraction(FImage image){

        final List<LocalFeature<SpatialLocation, FloatFV>> patchList = new ArrayList<>();
        RectangleSampler rec = new RectangleSampler(image.normalise(), 4, 4, 8, 8);

        for(Rectangle rectangle : rec.allRectangles()){
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



    // Hard assigner from Chapter 12
    static HardAssigner<float[], float[], IntFloatPair> trainQuantiliser(Dataset<FImage> sample){
        List<float[]> allKeys = new ArrayList<>();

        for(FImage img : sample){
            FImage image = img.getImage();
            // Getting a sample of patches
            List<LocalFeature<SpatialLocation, FloatFV>> samplePatches = getRandomElements(patchExtraction(image), 15);
            // Putting the vector values of the feature vectors in a list
            for(LocalFeature<SpatialLocation, FloatFV> localFeature : samplePatches){
                allKeys.add(localFeature.getFeatureVector().values);
            }
        }
        if(allKeys.size() > 10000) allKeys = allKeys.subList(0,10000);

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);

        float[][] dataSource = new float[allKeys.size()][];

        for(int i = 0; i < allKeys.size(); i++){
            dataSource[i] = allKeys.get(i);
        }

        FloatCentroidsResult result = km.cluster(dataSource);
        return result.defaultHardAssigner();
    }


    // Method to get random patches form the feature list
    static List<LocalFeature<SpatialLocation, FloatFV>> getRandomElements(List<LocalFeature<SpatialLocation, FloatFV>> patches, int numElements){
        List<LocalFeature<SpatialLocation, FloatFV>> randomList = new ArrayList<>();

        Random rand = new Random();

        for(int i = 0; i < numElements; i++){
            int index = rand.nextInt();

            randomList.add(patches.get(index));
        }

        return randomList;
    }
}

