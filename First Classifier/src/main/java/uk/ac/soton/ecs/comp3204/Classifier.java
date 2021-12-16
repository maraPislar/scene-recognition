package uk.ac.soton.ecs.comp3204;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

import java.lang.reflect.Field;

public interface Classifier {
    void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingImages);

    String classify(FImage image);
}
