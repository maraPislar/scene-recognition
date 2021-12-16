package uk.ac.soton.ecs.comp3204;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> images = new VFSGroupDataset<FImage>(
                "C:/Users/prana/University/Computer Vision/Group Project/training/training",
                ImageUtilities.FIMAGE_READER);

        Classifier knnClassifier = new TinyImageKNNClassifier(5);
        Classifier bagOfWordsClassifier = new BagOfWordsClassifier();
        //System.out.println(testClassifier(knnClassifier, images));
        System.out.println(testClassifier(bagOfWordsClassifier, images));
    }

    private static double testClassifier(Classifier classifier, VFSGroupDataset<FImage> images) {
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(images, 80, 0, 20);
        classifier.train(splits.getTrainingDataset());

        AtomicInteger correct = new AtomicInteger();
        AtomicInteger total = new AtomicInteger();
        splits.getTestDataset().getGroups().forEach(group ->
                splits.getTestDataset().get(group).forEach(testingImage -> {
                            if(classifier.classify(testingImage).equals(group)) {
                                correct.getAndIncrement();
                            }
                            total.getAndIncrement();
                        }
                ));

        return correct.doubleValue() / total.doubleValue();
    }



}
