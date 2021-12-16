package uk.ac.soton.ac.uk;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.math.geometry.shape.Rectangle;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class App {
    public static void main( String[] args ) throws FileSystemException {

        //Loading the dataset
        VFSGroupDataset<FImage> images = new VFSGroupDataset<>("src/main/resources/training", ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> testing = new VFSGroupDataset<>("testing", ImageUtilities.FIMAGE_READER);

        List<Rectangle> rectangleList = new ArrayList<>();
        Map<FImage, List<Rectangle>> imageRecMap = new HashMap<>();

        //Patches from the images
        for (FImage im : images) {
            //takes the image and the window information
            RectangleSampler rec = new RectangleSampler(im.normalise(), 4, 4, 8, 8);
            //rec.allRectangles().get(0).
            imageRecMap.put(im, rec.allRectangles());
        }
    }

    /**
     * Extracts the first 10000 dense SIFT features from the images in the dataset and then clusters them into
     * 500 separate classes
     * @param sample
     * @param pdsift
     * @return a HardAssigner which can be used to assign SIFT features to identifiers
     */
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiliser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift){
        List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList<>();
        for(FImage rec : sample){
            FImage image = rec.getImage();
            pdsift.analyseImage(image);
            allKeys.add(pdsift.getByteKeypoints(0.005f));
        }
        if(allKeys.size() > 10000) allKeys = allKeys.subList(0,10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
        DataSource<byte[]> dataSource = new LocalFeatureListDataSource<>(allKeys);
        ByteCentroidsResult result = km.cluster(dataSource);

        return result.defaultHardAssigner();
    }

    /**
     * BlockSpatialAggregator used with BagOfVisualWords to compute 4 histograms across the image
     */
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>{

        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner){
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bagOfVisualWords = new BagOfVisualWords<>(assigner);
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bagOfVisualWords, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}

