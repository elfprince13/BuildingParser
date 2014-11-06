package edu.brown.cs.buildingparser.detection

import org.opencv.objdetect.HOGDescriptor
import org.opencv.ml.CvSVM
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfFloat
import edu.brown.cs.buildingparser.Util

class SVMDetector(svmName:String, hogsUsed:List[HOGDescriptor]) {
	val hogBuckets = (hogsUsed.seq.map(hog => hog.get_winSize() -> hog)).toMap
	val biggestBucket = hogBuckets.keys.maxBy(_.area)
	
	val mySVM = new CvSVM()
	mySVM.load(svmName)
	
	def detect(inMat:Mat) = {
		val maxHog = hogBuckets(biggestBucket)
		val responses = hogsUsed.seq.map{hog => 
			val smallMat = new Mat
			val hogF = new MatOfFloat
			Imgproc.resize(inMat, smallMat, hog.get_winSize)
			hog.compute(smallMat, hogF)
			(hog.get_winSize -> mySVM.predict(Util.svmFeatureFromHog(hog, maxHog, hogF)))
		}.toMap
		responses.foreach{
			case(size,response) =>
				Console.println(size + ": " + response)
		}
		
		
	}
}