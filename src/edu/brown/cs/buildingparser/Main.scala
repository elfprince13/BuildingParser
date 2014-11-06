package edu.brown.cs.buildingparser

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.highgui.Highgui
import org.opencv.objdetect.HOGDescriptor
import org.opencv.core.MatOfFloat
import edu.brown.cs.buildingparser.training.ExtractSamples
import org.opencv.core.Scalar
import org.opencv.core.Size
import edu.brown.cs.buildingparser.training.SamplerMain
import edu.brown.cs.buildingparser.training.LearnObjectType
import edu.brown.cs.buildingparser.detection.SVMDetector
import edu.brown.cs.buildingparser.training.LearnerFactory

object Main {
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		val image = Highgui.imread(args(0), Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		
		val dimBuckets = SamplerMain.dimBuckets
		
		val boundBuckets = SamplerMain.boundBuckets
		
		//Util.makeImageFrame(Util.matToImage(image))
		
		val defaultHog = new HOGDescriptor
		val usedHogs = boundBuckets.map(windowSize => 
			new HOGDescriptor(windowSize, defaultHog.get_blockSize, defaultHog.get_blockStride, 
					defaultHog.get_cellSize, defaultHog.get_nbins))
		
		//val hogF = new MatOfFloat
		
		//hog.compute(image.submat(0,128,0,64), hogF)
		//Console.println(hogF.rows + " " + hogF.cols)
		//Console.println(hog.get_winSize)
		//Console.println(hog.getDescriptorSize() + " ");
		//Util.makeImageFrame(Util.matToImage(Util.visHogF(image, hog, hogF).submat(0,128*4,0,64*4)))
		//Util.svmFeatureFromHog(hog, maxHog, hogF)
		
		//Util.makeImageFrame(Util.matToImage(Util.makeBoundaryMirrored(image, 128)))
					
		/*
		val lot = new LearnObjectType("door","object-classes","anti-object-classes",
				List("window","balcony"),Set[String](".png",".jpg",".jpeg"), usedHogs)
		lot.initExamplesLists
		val testImg = Highgui.imread(lot.posExamples(0), Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		Console.println("Testing: " + testImg.size)
		val sd = new SVMDetector("trained-svms/1415252858024_door_.opencv_svm",usedHogs)
		sd.detect(testImg)
		val testImg2 = Highgui.imread(lot.negExamples(lot.negExamples.size - 35), Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		Console.println("Testing: " + testImg2.size)
		sd.detect(testImg2)
		*/
		
		
	}
}