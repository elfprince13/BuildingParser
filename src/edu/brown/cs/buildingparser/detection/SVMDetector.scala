package edu.brown.cs.buildingparser.detection

import org.opencv.objdetect.HOGDescriptor
import org.opencv.ml.CvSVM
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfFloat
import edu.brown.cs.buildingparser.Util
import org.opencv.core.Size
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Point
import scala.collection.SeqView
import org.apache.commons.io.IOUtils
import scala.util.Marshal

class SVMDetector(svmName:String, hogsUsed:List[HOGDescriptor], histogramName:String = "", histogramThreshold:Int = 1) {
	val histogram = if(histogramName != ""){
		Console.println("Restricting hogsUsed by example-histogram with threshold: " + histogramThreshold)
		val in = new java.io.FileInputStream(histogramName)
		Marshal.load[Map[(Double,Double),Int]](IOUtils.toByteArray(in)).map(p => new Size(p._1._1, p._1._2) -> p._2)
	} else { null }
	val hogBuckets = (hogsUsed.view.filter(hog => (histogram == null) || 
			(histogram.getOrElse(hog.get_winSize,0) > histogramThreshold)).map{hog => 
				if(histogram != null){
					Console.println("Histogram allowed detector for " + hog.get_winSize + " ( " + histogram(hog.get_winSize) + " > " + histogramThreshold + " )")
				}
				hog.get_winSize -> hog}).toMap
	val maxHog = hogsUsed.maxBy(_.get_winSize.area) // Can't use buckets directly, since we might not have been invited to the party
	val DETECTION_THRESHOLD = 0.5
	
	val mySVM = new CvSVM()
	mySVM.load(svmName)
	
	def detect(inMat:Mat) = {
		val responses = hogBuckets.view.map{case(hogSize, hog) => 
			//val smallMat = new Mat
			val hogF = new MatOfFloat
			val locs = new MatOfPoint
			//Imgproc.resize(inMat, smallMat, hog.get_winSize)
			val imgSize = inMat.size
			val borderSize = new Size(hogSize.width / 2, hogSize.height / 2)
			val cellSize = hog.get_cellSize
			// might be off by one
			val widthInCells = ((imgSize.width + cellSize.width).intValue / cellSize.width.intValue)
			val heightInCells = ((imgSize.height + cellSize.height).intValue / cellSize.height.intValue)
			hog.compute(inMat, hogF, cellSize, borderSize, locs)
			val hogLen = hog.getDescriptorSize.intValue
			val timesComputed = hogF.size.height / hogLen
			if (widthInCells * heightInCells != timesComputed){
				throw new IllegalStateException("the sliding window calculation doesn't check out")
			}
			Console.println( hogSize + " " + cellSize + " " + borderSize + " " + hogF.size + " " + hogLen + " " + timesComputed)
			(0 until timesComputed.intValue).view.map{
				winNum =>
				val predicted = mySVM.predict(Util.svmFeatureFromHog(hog, maxHog, hogF.submat(winNum*hogLen,(winNum+1)*hogLen,0,1)))
				val x = (winNum % widthInCells) * cellSize.width
				val y = (winNum / widthInCells) * cellSize.height
				/*if(predicted != 0){
					Console.println("hit: " + x + " " + y + " ( " + predicted + " )")
				}*/
				(new Rect(new Point(x,y), hogSize) -> predicted)
			}.filter(_._2 > DETECTION_THRESHOLD )
		}.flatten
		
		responses
		
	}
}