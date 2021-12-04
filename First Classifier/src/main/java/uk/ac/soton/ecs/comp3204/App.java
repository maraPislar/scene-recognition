package uk.ac.soton.ecs.comp3204;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.array.ArrayUtils;

import java.util.*;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> trainingImages = new VFSGroupDataset<FImage>(
                "C:/Users/prana/University/Computer Vision/Group Project/training/training",
                ImageUtilities.FIMAGE_READER);
        System.out.println(trainingImages.size());

        TinyImageKNNClassifier classifier = new TinyImageKNNClassifier(trainingImages);

        VFSListDataset<FImage> testingImages = new VFSListDataset<FImage>(
                "C:/Users/prana/University/Computer Vision/Group Project/testing/testing",
                ImageUtilities.FIMAGE_READER);

        Map<String, String> resultMap = new HashMap<>();
        for(int i = 0; i < testingImages.size(); i++) {
            resultMap.put(testingImages.getID(i), classifier.classify(testingImages.get(i), 5));
        }

        String[] sortedResults = new String[resultMap.size()];
        int j = 0;
        for(Map.Entry<String, String> result : resultMap.entrySet()) {
            sortedResults[j] = result.toString();
            j++;
        }

        Arrays.sort(sortedResults);
        Arrays.stream(sortedResults).forEach(System.out::println);


        System.out.println(resultMap.size());

    }



}
