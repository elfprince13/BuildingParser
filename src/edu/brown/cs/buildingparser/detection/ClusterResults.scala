package edu.brown.cs.buildingparser.detection

import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.objdetect.HOGDescriptor
import edu.brown.cs.buildingparser.Util
import org.opencv.core.MatOfFloat
import jsat.clustering.MeanShift
import jsat.SimpleDataSet
import jsat.linear.DenseVector
import jsat.classifiers.DataPoint
import jsat.classifiers.CategoricalData

import collection.JavaConverters._

class FindObjectKinds(img:Mat) {
	
	def makeClustersOf(finds:List[(Rect,Mat)], usedHogs:List[HOGDescriptor]) = {
		val hogBuckets = usedHogs.map{hog => hog.get_winSize -> hog}.toMap
		val maxHog = usedHogs.maxBy(_.get_winSize.area) // Can't use buckets directly, since we might not have been invited to the party
		val featureSpace = finds.map{
			case(box,subImg) => 
				val hog = hogBuckets(subImg.size)
				val hogF = new MatOfFloat
				hog.compute(subImg,hogF)
				val outF = Util.svmFeatureFromHog(hog, maxHog, hogF)
				Console.println(outF.dump())
				outF
		}
	
		Console.println("Clustering found features")
		val clusterer = new MeanShift
		val data = new SimpleDataSet(featureSpace.map{feat => 
		val rowArray = new Array[Double](feat.width)
		(0 until rowArray.length).foreach{ i=>
			rowArray(i) = feat.get(0, i)(0)
		}
		val rectV = new DenseVector(rowArray)
		val dp = new DataPoint(rectV, new Array[Int](0), new Array[CategoricalData](0))
		dp
		}.asJava)
	
		val clusterAssignments = clusterer.cluster(data, null)
		clusterAssignments.foreach{
		i => Console.print(i + " ")
		}
		Console.println("")
		clusterAssignments
	}
}