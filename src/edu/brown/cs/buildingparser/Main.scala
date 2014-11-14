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
import jsat.linear.DenseVector
import collection.JavaConverters._
import scala.util.Random
import jsat.SimpleDataSet
import jsat.classifiers.DataPoint
import jsat.classifiers.CategoricalData
import jsat.clustering.MeanShift
import org.opencv.imgproc.Imgproc
import org.opencv.core.Rect
import org.opencv.core.Point
import edu.brown.cs.buildingparser.synth.LDrawGridify
import edu.brown.cs.buildingparser.synth.ObjConstraints

object Main {
	def jDL(l:List[Double]):java.util.List[java.lang.Double] = {
		l.map(_.asInstanceOf[java.lang.Double]).asJava
	}
	
	def ofsPoint(pt:Point, ofs:Point):Point = {
		new Point(pt.x + ofs.x, pt.y + ofs.y)
	}
	
	def showObjectBorders(image:Mat, contents:Map[String, List[(Rect,Mat)]]):Unit = {
		contents.foreach{
			case(labelName, objs) => objs.foreach{
					case(box, subImg) => 
						Core.rectangle(image, box.tl, box.br, SamplerMain.stdLabelProps(labelName)._1)
				}
		}
	}
	
	def showAvgObjects(image:Mat, clusteredContents:Map[String, Map[Int,(Mat,List[Rect])]], ofs:Point) = {
		clusteredContents.foreach{
			case(labelName, clusters) =>
				clusters.foreach{
					case(clusterNum, cluster) => 
						val repM = cluster._1
						cluster._2.foreach{
							rect => 
								val ofsBox = new Rect(ofsPoint(rect.tl, ofs), rect.size)
								repM.copyTo(image.submat(ofsBox))
						}
				}		
		}

	}
	
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		SamplerMain.isMain = false
		val dimBuckets = SamplerMain.dimBuckets
		val boundBuckets = SamplerMain.boundBuckets
		
		val clusterCenters = List(
				new DenseVector(jDL(List(1,0,0))), new DenseVector(jDL(List(0,0,0))), 
				new DenseVector(jDL(List(1,0,1))), new DenseVector(jDL(List(0,0,1))), 
				new DenseVector(jDL(List(1,1,0))), new DenseVector(jDL(List(0,1,0))), 
				new DenseVector(jDL(List(1,1,1))), new DenseVector(jDL(List(0,1,1))))
				
		val rg = new Random
		val dvs = clusterCenters.view.map(dv => Seq.fill(5)(dv.add(new DenseVector(jDL(Seq.fill(3)(rg.nextDouble / 20).toList))))).flatten.toList
		val data = new SimpleDataSet(dvs.map(dv => new DataPoint(dv, new Array[Int](0), new Array[CategoricalData](0))).asJava)
		val clusterer = new MeanShift
		//Util.dumpClusters(clusterer.cluster(data).asScala.map(_.asScala.toList).toList)
		
		
		
		val defaultHog = new HOGDescriptor
		val usedHogs = boundBuckets.map(windowSize => 
			new HOGDescriptor(windowSize, defaultHog.get_blockSize, defaultHog.get_blockStride, 
					defaultHog.get_cellSize, defaultHog.get_nbins))
		
		val grazSampler = SamplerMain.grazSampler
		val srcHandle = grazSampler.imgHandleFromName("facade_0_0099003_0099285.png")
		val labelHandle = grazSampler.pairedImgs(srcHandle)
		val (srcBase, examples) = grazSampler.extractOneExampleSet(srcHandle, labelHandle)
		val srcImg = Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)
		val image = Util.makeBoundaryMirrored(srcImg, 128)
		val cr = new FindObjectKinds(image)
		val imgContents = examples.view/*.filter(_._1 == "window")*/.map{
			case(labelName, labelColor:Scalar, posStream, _) =>
				val objs = posStream.map(found => (found._1,found._3)).toList
				Console.println("Have " + objs.length + " " + labelName + "s")
				(labelName -> objs)
		}.toMap
		val clusteredContents = cr.clusterAllObjects(imgContents, usedHogs).map{
			case(labelName, clusters) =>
				(labelName -> clusters.map{
					case(clusterNum, cluster) => 
						(clusterNum, (cluster._1, cluster._2.map(
							rect => Util.calcGrabBox(rect, Util.bestBoundsBucket(rect, boundBuckets))
						)))
				})		
		}
		
		showAvgObjects(image, clusteredContents, new Point(0, 0))
		showObjectBorders(image, imgContents)
		
		
		val imgClip = image.submat(128, image.rows - 128, 128, image.cols - 128) 
		Util.makeImageFrame(Util.matToImage(imgClip))
		Util.makeImageFrame(Util.matToImage(srcImg))
		val imgDiff = new Mat
		Core.absdiff(srcImg, imgClip, imgDiff)
		Util.makeImageFrame(Util.matToImage(imgDiff))
		
		val borderOfs = new Point(-128, -128)
		val boxesImg = Mat.zeros(srcImg.size, srcImg.`type`)
		//val griddedBoxesImg = Mat.zeros(LDrawGridify.snapSizeToGrid(srcImg.size), srcImg.`type`)
		val griddedBounds = new Size(
				LDrawGridify.pixelsToGridUnits(srcImg.size.width.intValue),
				LDrawGridify.pixelsToGridUnits(srcImg.size.height.intValue)
				)
		val griddedBoxes = clusteredContents.map{
			case(labelName, clusters) =>
				(labelName -> clusters.map{
					case(clustNum, cluster) =>
						(clustNum -> cluster._2.map{
							box => 
								val grabBox = Util.calcGrabBox(box, Util.bestBoundsBucket(box, boundBuckets))
								val fixPt = ofsPoint(grabBox.tl, borderOfs)
								val fixSz = grabBox.size
								val nx = LDrawGridify.pixelsToGridUnits(fixPt.x.intValue)
								val ny = LDrawGridify.pixelsToGridUnits(fixPt.y.intValue)
								val nw = LDrawGridify.pixelsToGridUnits(fixSz.width.intValue)
								val nh = LDrawGridify.pixelsToGridUnits(fixSz.height.intValue)
									
								new Rect(new Point(nx, ny), new Size(nw, nh))
						})
				})
		}
		
		val solver = new ObjConstraints(griddedBounds, griddedBoxes, LDrawGridify.gridStep)
		val stats = solver.trySolve(runs = 300, fails = 600, prob = 80)
		//Console.println(stats)
		if(solver.solved){
			showObjectBorders(boxesImg, imgContents.map{
				case(labelName, boxes) =>
					(labelName -> boxes.map{
						case(box, obj) => 
							val grabBox = Util.calcGrabBox(box, Util.bestBoundsBucket(box, boundBuckets))
							(new Rect(ofsPoint(grabBox.tl, borderOfs), grabBox.size), obj)
					})
			})
			
			val griddedBoxesImg = Mat.zeros(solver.getSolvedBoundary, boxesImg.`type`)
			val griddedClusteredContents = solver.getSolvedObjs
			val griddedContents = griddedClusteredContents.map{
				case(labelName, clusters) =>
					(labelName, clusters.view.map{
						case(clustNum, boxes) => boxes.view.map(box => (box, new Mat))
					}.flatten.toList)
			}
			
			showObjectBorders(griddedBoxesImg, griddedContents)
			
			Util.makeImageFrame(Util.matToImage(boxesImg), "boxes")
			Util.makeImageFrame(Util.matToImage(griddedBoxesImg), "gridded boxes")
			
		} else {
			Console.println("Could not achieve a valid gridded facade.")
			Console.println("\t=> Try adjusting the mesh resolution or deletion weights")
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
		
		

		//LearnerFactory.learnAll(usedHogs)
	}
}