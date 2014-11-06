package edu.brown.cs.buildingparser

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.highgui.Highgui
import org.opencv.objdetect.HOGDescriptor
import org.opencv.core.MatOfFloat
import edu.brown.cs.buildingparser.ui.Util
import edu.brown.cs.buildingparser.training.ExtractSamples
import org.opencv.core.Scalar
import org.opencv.core.Size

object Main {
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		val image = Highgui.imread(args(0), Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		
		val dimBuckets = List(16, 32, 48, 64, 96, 128)
		
		val boundBuckets = (for( i <- dimBuckets; j <- dimBuckets) yield Pair(i,j)) map {
			case (i, j) => new Size(i, j)
		}
		
		//Util.makeImageFrame(Util.matToImage(image))
		
		val hog = new HOGDescriptor
		val maxHog = new HOGDescriptor(new Size(128, 128), hog.get_blockSize, hog.get_blockStride, hog.get_cellSize, hog.get_nbins)
		
		val hogF = new MatOfFloat
		
		//hog.compute(image.submat(0,128,0,64), hogF)
		//Console.println(hogF.rows + " " + hogF.cols)
		//Console.println(hog.get_winSize)
		//Console.println(hog.getDescriptorSize() + " ");
		//Util.makeImageFrame(Util.matToImage(Util.visHogF(image, hog, hogF).submat(0,128*4,0,64*4)))
		//Util.svmFeatureFromHog(hog, maxHog, hogF)
		
		//Util.makeImageFrame(Util.matToImage(Util.makeBoundaryMirrored(image, 128)))
		
		
		/*
		val dummySampler = new ExtractSamples(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/dummy",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/dummy",
				"object-classes", Set[String](".png",".jpg",".jpeg"),
				hog.get_winSize, Map[String,(Scalar,Boolean)]("window" -> (new Scalar(0,0,255),true)), Map[(Scalar,Boolean),String]((new Scalar(0,0,255),true) -> "window")
		)
		//*/

		/*
		val parisSampler = new ExtractSamples(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/cvpr2010/images",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/ground_truth_2011.zip Folder",
				"object-classes", Set[String](".png",".jpg",".jpeg"),
				hog.get_winSize, Map[String,(Scalar,Boolean)](//"window" -> (new Scalar(0,0,255),true),
			 	//"balcony" -> (new Scalar(255,0,128),true),
			 	"door" -> (new Scalar(0,128,255),true)), Map[(Scalar,Boolean),String](//(new Scalar(0,0,255),true) -> "window",
			 	//(new Scalar(255,0,128),true) -> "balcony",
			 	(new Scalar(0,128,255),true) -> "door")
		)
		//*/
		
		/*
		val grazSampler = new ExtractSamples(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/graz50_facade_dataset.zip Folder/graz50_facade_dataset/images",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/graz50_facade_dataset.zip Folder/graz50_facade_dataset/labels_full",
				"object-classes", Set[String](".png",".jpg",".jpeg"),
				hog.get_winSize, Map[String,(Scalar,Boolean)](//"window" -> (new Scalar(0,0,255),true),
			 	//"balcony" -> (new Scalar(255,0,128),true),
			 	"door" -> (new Scalar(0,128,255),true)
				), Map[(Scalar,Boolean),String](//(new Scalar(0,0,255),true) -> "window",
			 	//(new Scalar(255,0,128),true) -> "balcony",
			 	(new Scalar(0,128,255),true) -> "door")
		
		)
		//*/
		
		//*
		val grazSampler = new ExtractSamples(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/graz-small",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionandGraphics/graz50_facade_dataset.zip Folder/graz50_facade_dataset/labels_full",
				"object-classes", "anti-object-classes", Set[String](".png",".jpg",".jpeg"),
				boundBuckets, Map[String,(Scalar,Boolean)]("window" -> (new Scalar(0,0,255),true),
			 	//"balcony" -> (new Scalar(255,0,128),true),
			 	"door" -> (new Scalar(0,128,255),true)
				), Map[(Scalar,Boolean),String]((new Scalar(0,0,255),true) -> "window",
			 	//(new Scalar(255,0,128),true) -> "balcony",
			 	(new Scalar(0,128,255),true) -> "door")
		
		)
		//*/
		
	}
}