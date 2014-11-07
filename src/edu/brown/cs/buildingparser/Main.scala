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
import edu.brown.cs.buildingparser.detection.FindObjectKinds

object Main {
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		SamplerMain.isMain = false
		val dimBuckets = SamplerMain.dimBuckets
		val boundBuckets = SamplerMain.boundBuckets
		
		
		val defaultHog = new HOGDescriptor
		val usedHogs = boundBuckets.map(windowSize => 
			new HOGDescriptor(windowSize, defaultHog.get_blockSize, defaultHog.get_blockStride, 
					defaultHog.get_cellSize, defaultHog.get_nbins))
		
		val grazSampler = SamplerMain.grazSampler
		val srcHandle = grazSampler.imgHandleFromName("facade_1_0056092_0056345.png")
		val labelHandle = grazSampler.pairedImgs(srcHandle)
		val (srcBase, examples) = grazSampler.extractOneExampleSet(srcHandle, labelHandle)
		val image = Util.makeBoundaryMirrored(Highgui.imread(srcHandle.getAbsolutePath), 128)
		val cr = new FindObjectKinds(image)
		examples.filter(_._1 == "window").toList(0) match {
			case(_, labelColor:Scalar, posStream, _) =>
				val windows = posStream.map(found => (found._1,found._3)).toList
				Console.println("Have " + windows.length + " windows")
				cr.makeClustersOf(windows, usedHogs)
				windows.foreach{
					case(box, subImg) => 
						Core.rectangle(image, box.tl, box.br, new Scalar(1.0))
				}
		}
		
		
		/*
		val image = Highgui.imread(args(0), Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		Console.println("Testing: " + image.size)
		val wd = new SVMDetector("trained-svms/1415255461043_window_.opencv_svm",usedHogs,"trained-svms/window-posExSizes.histogram")
		val detected = wd.detect(image).map{case(rect, response) => rect}.toList
		val assignedClusters = Util.clusterRects(detected)
		*/
		
		/*
		foreach{
			case(found, response) =>
				Core.rectangle(image, found.tl, found.br, new Scalar(1.0))
		}
		
		*/
		Util.makeImageFrame(Util.matToImage(image))
		

		//LearnerFactory.learnAll(usedHogs)
	}
}