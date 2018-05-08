package me.planer.mlplayground.doodledetect;

import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.event.KeyEvent;
import processing.event.MouseEvent;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main extends PApplet {

    MultiLayerNetwork model;
    List<String> classes;
    public static void main(String[] args) {
        PApplet.main(Main.class.getCanonicalName());
    }

    @Override
    public void setup() {
        try {
            URL url = Main.class.getResource("/model_cnn.hdf5");
            model = KerasModelImport.importKerasSequentialModelAndWeights(url.getPath());
            classes = new ArrayList<>(Arrays.asList("apple", "banana", "bird", "cat", "hat", "shark", "star", "table", "truck"));
        } catch (Exception e) {
            e.printStackTrace();
            exit();
        }

        background(255);
    }

    @Override
    public void draw() {
    }

    @Override
    public void settings() {
        size(600, 600);
    }

    @Override
    public void mouseDragged(MouseEvent event) {
        if (event.getButton() == RIGHT) {
            stroke(255);
        } else {
            stroke(0, 0, 255);
        }
        strokeWeight(12.0f);
        line(pmouseX, pmouseY, mouseX, mouseY);
        surface.setTitle("Drawing...");
    }

    @Override
    public void keyPressed(KeyEvent event) {
        if (event.getKey() == 'x') {
            background(255);
        } else if (event.getKey() == 'c') {
            predict();
        } else if (event.getKey() == 'b') {
            Rect box = boundingBox();
            stroke(0);
            strokeWeight(1);
            noFill();
            rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            println("Bounding box:", box.x1, box.y1, box.x2, box.y2);
        }
    }

    @Override
    public void mouseReleased() {
        predict();
    }

    private Rect boundingBox() {
        int minX = width - 1;
        int minY = height - 1;
        int maxX = 0;
        int maxY = 0;

        loadPixels();
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int i = y * width + x;

                int pixel = pixels[i];
                if (pixel == 255 || pixel == -1) continue;

                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }

        int centerX = (minX + maxX) / 2;
        int centerY = (minY + maxY) / 2;
        int longestSide = max(maxX - minX, maxY - minY);

        minX = centerX - longestSide / 2;
        maxX = centerX + longestSide / 2;
        minY = centerY - longestSide / 2;
        maxY = centerY + longestSide / 2;

        minX = constrain(minX, 0, width - 1);
        maxX = constrain(maxX, 0, width - 1);
        minY = constrain(minY, 0, height - 1);
        maxY = constrain(maxY, 0, height - 1);

        if (minX >= maxX || minY >= maxY) return new Rect(0, 0, width - 1, height - 1);


        return new Rect(minX, minY, maxX, maxY);
    }

    private void predict() {
        int size = 28;
        Rect box = boundingBox();
        PImage pImage = get(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        pImage.filter(PConstants.GRAY);
        pImage.filter(PConstants.INVERT);
        pImage.resize(size, size);

        BufferedImage image = toBufferedImage(pImage.getImage());
        INDArray ndArray = arrayFromImage(image);

        int[] output = model.predict(ndArray);
        String classPrediction = classes.get(output[0]);
        System.out.println(classPrediction);
        surface.setTitle("Prediction: " + classPrediction);
    }

    public INDArray arrayFromImage(BufferedImage image) {
        ImageLoader loader = new ImageLoader(28, 28, 1, true);
        INDArray ndArray = loader.toBgr(image);

        return ndArray.reshape(1, 1, 28, 28);
    }


    private BufferedImage toBufferedImage(Image img) {
        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_BYTE_GRAY);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

    private class Rect {
        int x1;
        int y1;
        int x2;
        int y2;

        public Rect(int x1, int y1, int x2, int y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
    }

}
