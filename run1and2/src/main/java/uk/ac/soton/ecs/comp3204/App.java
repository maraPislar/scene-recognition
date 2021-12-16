package uk.ac.soton.ecs.comp3204;

import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws IOException {
        VFSGroupDataset<FImage> trainingImages = new VFSGroupDataset<FImage>(
                "C:/Users/prana/University/Computer Vision/Group Project/training/training",
                ImageUtilities.FIMAGE_READER);

        Classifier knnClassifier = new TinyImageKNNClassifier(5);
        Classifier bagOfWordsClassifier = new BagOfWordsClassifier();

        VFSListDataset<FImage> testingImages = new VFSListDataset<FImage>(
                "C:/Users/prana/University/Computer Vision/Group Project/testing/testing",
                ImageUtilities.FIMAGE_READER);

        //outputResults(knnClassifier, trainingImages, testingImages, "run1.txt");
        //outputResults(bagOfWordsClassifier, trainingImages, testingImages, "run2.txt");

        System.out.println(testClassifier(knnClassifier, trainingImages));
        //System.out.println(testClassifier(bagOfWordsClassifier, trainingImages));
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
            })
        );

        return correct.doubleValue() / total.doubleValue();
    }

    private static void outputResults(Classifier classifier, VFSGroupDataset<FImage> trainingImages, VFSListDataset<FImage> testingImages, String filename) throws IOException {
        GroupedDataset<String, ListDataset<FImage>, FImage> convertedTrainingImages = new MapBackedDataset<>();
        convertedTrainingImages.putAll(trainingImages);
        classifier.train(convertedTrainingImages);

        String[] results = new String[testingImages.size()];
        for(int i = 0; i < testingImages.size(); i++) {
            results[i] = testingImages.getID(i) + " " + classifier.classify(testingImages.get(i));
        }

        Arrays.sort(results, (a, b) ->
            Integer.parseInt(a.split("\\.")[0]) > Integer.parseInt(b.split("\\.")[0]) ? 1 : -1
        );

        FileWriter writer = new FileWriter("C:/Users/prana/University/Computer Vision/Group Project/" + filename);
        for (String result : results) {
            writer.write(result + "\n");
        }
        writer.close();

        //Arrays.stream(results).forEach(System.out::println);
    }

}
