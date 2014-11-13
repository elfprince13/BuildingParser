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
import org.opencv.imgproc.Imgproc
import org.opencv.core.Size
import org.opencv.core.CvType
import org.opencv.highgui.Highgui
import jsat.distributions.multivariate.MetricKDE
import jsat.linear.distancemetrics.EuclideanDistance
import jsat.distributions.empirical.kernelfunc.GaussKF
import org.opencv.core.Core
import org.opencv.core.Scalar

class FindObjectKinds(img:Mat) {
	
	def clusterAllObjects(allObjs:Map[String,List[(Rect,Mat)]], usedHogs:List[HOGDescriptor]):Map[String, Map[Int,(Mat,List[Rect])]] = {
		allObjs.map{
			case(label, objs) =>
				val clusters = makeClustersOf(objs, usedHogs)
				(label -> clusters)		
		}
		
	}
	
	def makeClustersOf(finds:List[(Rect,Mat)], usedHogs:List[HOGDescriptor]):Map[Int,(Mat,List[Rect])] = {
		val hogBuckets = usedHogs.map{hog => hog.get_winSize -> hog}.toMap
		val maxHog = usedHogs.maxBy(_.get_winSize.area) // Can't use buckets directly, since we might not have been invited to the party
		if(finds.size > 1){
			val featureSpace = finds.map{
				case(box,subImg) => 
					val hog = hogBuckets(subImg.size)
					val hogF = new MatOfFloat
					hog.compute(subImg,hogF)
					//Console.println(hogF.t().dump())
					//hogF.t()
					/*
					val outF = new Mat
					Imgproc.resize(subImg, outF, new Size(16, 16)) 
					Console.println(outF.reshape(1,1).dump())
					outF.reshape(1, 1)
					*/
					Util.svmFeatureFromHog(hog, maxHog, hogF)
					
			}
			Console.println(featureSpace(0).width)
			
			/*
			val compImg = new Mat(featureSpace.size, featureSpace(0).width, CvType.CV_8U)
			val featScale = 1
			featureSpace.zipWithIndex.foreach{
				case(row, i)=> 
					(0 until row.width).foreach{ j=>
						val outInt = (row.get(0,j)(0)*featScale).byteValue
						val outArray = new Array[Byte](1)
						outArray(0) = outInt
						compImg.put(i, j, outArray)
						//row.copyTo(compImg.submat(i, i+1, 0, 540))	
					}
					
			}
			//Util.makeImageFrame(Util.matToImage(compImg))
			//Highgui.imwrite("features-hog.png", compImg)
			*/
			
		
			Console.println("Clustering found-features")
			val data = new SimpleDataSet(featureSpace.map{feat => 
			val rowArray = new Array[Double](feat.width)
			(0 until rowArray.length).foreach{ i=>
				rowArray(i) = feat.get(0, i)(0)
			}
			val rectV = new DenseVector(rowArray)
			val dp = new DataPoint(rectV, new Array[Int](0), new Array[CategoricalData](0))
			dp
			}.asJava)
			
			val mkde = new MetricKDE(GaussKF.getInstance(), new EuclideanDistance)
			val bandwidth = Util.calcBandwidth(data, 0.0125)
			mkde.setBandwith(bandwidth)
			mkde.setDefaultK(1)
			Console.println("Bandwidth: " + mkde.getBandwith())
			val clusterer = new MeanShift(mkde)
			val clusterAssignments = clusterer.cluster(data,null)
			clusterAssignments.foreach{ cv =>
				Console.print(cv + ", ")
			}
			Console.println("")
			
			val clusters:Map[Int,(Mat,List[Rect])] = (finds zip clusterAssignments).foldLeft(Map[Int,(Mat,List[Rect])]()){
				(resMap,inst) =>
					//Util.makeImageFrame(Util.matToImage(inst._1._2), "Cluster " + inst._2)
					val nM = new Mat(inst._1._2.size, CvType.CV_32FC(inst._1._2.channels))
					//val dummy = new Mat(inst._1._2.size, CvType.CV_8UC(inst._1._2.channels))
					inst._1._2.convertTo(nM, CvType.CV_32FC(inst._1._2.channels))
					val nR = inst._1._1
					val updCluster = if(resMap contains inst._2){
						val curPair = resMap(inst._2)
						val outM = new Mat
						Core.add(curPair._1, nM,outM)
					
						(outM, curPair._2 :+ nR)
					} else {
						(nM, List(nR))
					}
					resMap + (inst._2 -> updCluster)
			}.map{
				case(clusterNum,cluster) =>
					val fillVal = 1. / cluster._2.size
					val fill = new Array[Double](cluster._1.channels)
					(0 until fill.length).foreach{ i => fill(i) = fillVal}
					val clustScale = new Mat(cluster._1.size, CvType.CV_32FC(cluster._1.channels), new Scalar(fill))
					
					/*
					val dummy = new Mat(cluster._1.size, CvType.CV_8UC(cluster._1.channels))
					clustScale.convertTo(dummy, CvType.CV_8UC(clustScale.channels))
					Util.makeImageFrame(Util.matToImage(dummy), "Cluster " + clusterNum)
					*/
					
					Console.println(clustScale.size + " " + clustScale.`type` + " " + cluster._1.size + " " + cluster._1.`type` + " " + fillVal)
					val rescaled = cluster._1.mul(clustScale)
					val outM = new Mat
					rescaled.convertTo(outM, CvType.CV_8UC(cluster._1.channels))
					//Util.makeImageFrame(Util.matToImage(outM), "Cluster " + clusterNum)
					(clusterNum -> (outM, cluster._2))
			}
			//Util.dumpClusters(clusterAssignments.asScala.map(_.asScala.toList).toList)
			clusters
		} else if (finds.size == 1) {
			val obj = finds(0)
			val tinyCluster = (obj._2, List(obj._1))
			Map(0 ->  tinyCluster)
		} else {
			Map[Int,(Mat,List[Rect])]()
		}
	}
}