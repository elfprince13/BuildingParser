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
import edu.brown.cs.buildingparser.library.BrickSynth
import edu.brown.cs.buildingparser.synth.SimpleDragger
import edu.brown.cs.buildingparser.synth.ObjConstraints
import edu.brown.cs.buildingparser.synth.DPSubdivider
import edu.brown.cs.buildingparser.synth.DPEvaluator

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
						Core.rectangle(image, box.tl, ofsPoint(box.br, new Point(-1, -1)), SamplerMain.stdLabelProps(labelName)._1)
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
		
		val rg = new Random
		
		/*
		val clusterCenters = List(
				new DenseVector(jDL(List(1,0,0))), new DenseVector(jDL(List(0,0,0))), 
				new DenseVector(jDL(List(1,0,1))), new DenseVector(jDL(List(0,0,1))), 
				new DenseVector(jDL(List(1,1,0))), new DenseVector(jDL(List(0,1,0))), 
				new DenseVector(jDL(List(1,1,1))), new DenseVector(jDL(List(0,1,1))))
				
		val dvs = clusterCenters.view.map(dv => Seq.fill(5)(dv.add(new DenseVector(jDL(Seq.fill(3)(rg.nextDouble / 20).toList))))).flatten.toList
		val data = new SimpleDataSet(dvs.map(dv => new DataPoint(dv, new Array[Int](0), new Array[CategoricalData](0))).asJava)
		val clusterer = new MeanShift
		
		 //Util.dumpClusters(clusterer.cluster(data).asScala.map(_.asScala.toList).toList)
		 */
		
		
		
		val defaultHog = new HOGDescriptor
		val usedHogs = boundBuckets.map(windowSize => 
			new HOGDescriptor(windowSize, defaultHog.get_blockSize, defaultHog.get_blockStride, 
					defaultHog.get_cellSize, defaultHog.get_nbins))
		
		val grazSampler = SamplerMain.grazSampler
		val srcHandle = grazSampler.imgHandleFromName("facade_0_0053403_0053679.png")//("facade_1_0056092_0056345.png")//("facade_0_0099003_0099285.png")//
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
		//showObjectBorders(image, imgContents)
		
		
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
		val stats = solver.trySolve(runs = 20, fails = 600, prob = 80)
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
			Util.makeImageFrame(Util.matToImage(boxesImg), "boxes")
			
			
			val griddedBoxesImg = Mat.zeros(solver.getSolvedBoundary, boxesImg.`type`)
			val griddedClusteredContents = solver.getSolvedObjs
			val griddedContents = griddedClusteredContents.map{
				case(labelName, clusters) =>
					(labelName, clusters.view.map{
						case(clustNum, boxes) => boxes.view.map(box => (box, new Mat))
					}.flatten.toList)
			}
			
			showObjectBorders(griddedBoxesImg, griddedContents)
			Util.makeImageFrame(Util.matToImage(griddedBoxesImg), "gridded objects")
			
			
			val imgRemapped = Mat.zeros(griddedBoxesImg.size, griddedBoxesImg.`type`)
			val boxTargets = clusteredContents.map{
				case(clusterName, srcCluster) =>
					val dstCluster = griddedClusteredContents(clusterName)
					srcCluster.map{
						case(clustNum, srcRects) =>
							val dstRects = dstCluster(clustNum)
							dstRects.zip(srcRects._2.map{
								rect =>
									new Rect(ofsPoint(rect.tl, borderOfs), rect.size)
							})
					}.flatten
					
			}.flatten.toList
			
			val stripper = new DPSubdivider(LDrawGridify.gridStep)
			val boundaryRect = new Rect(new Point(0,0),solver.getSolvedBoundary)
			val regions = stripper.getNonObjRegions(boundaryRect, stripper.sortObjs(griddedClusteredContents.values.flatMap(_.values.flatten).toList))//, Some(griddedBoxesImg))
			Console.println(f"output image for boxes has dims ${griddedBoxesImg.width} x ${griddedBoxesImg.height}")
			regions.zipWithIndex.foreach{
				case(region, i) => 
					Console.println(f"Drawing region $i / ${regions.length}")
					val rn = rg.nextInt(256)
					val color = new Scalar(128 + rg.nextInt(128), 128 + rg.nextInt(128), rn)	
					Util.checkContains(boundaryRect, region) match {
						case None => Console.println("Skipping bad rectangle")
						case Some(region) =>
							Core.rectangle(griddedBoxesImg, region.tl, ofsPoint(region.br, new Point(-1, -1)), color, -1)
					}
			}
			Util.makeImageFrame(Util.matToImage(griddedBoxesImg), "gridded boxes")
			
			
			val dragger = new SimpleDragger(imgClip.size, imgRemapped.size, boxTargets)
			dragger.dragObjs(imgClip, imgRemapped)
			Util.makeImageFrame(Util.matToImage(imgRemapped),"Remapping result")
			val dpTarget = new Mat(imgRemapped.size, imgRemapped.`type`)
			
			val brickLib = BrickSynth.getStdBricks()
			val brickPlacer = new DPEvaluator(LDrawGridify.gridStep, l = 1.01, k = 1)//0.0125)
			Console.println(f"output image for render has dims ${dpTarget.width} x ${dpTarget.height}")
			regions.zipWithIndex.foreach{
				case(region, i) => 
					Console.println(f"Drawing region $i / ${regions.length}")
					// This is an ugly hack around the CP implementation lying to us
					Util.checkContains(boundaryRect, region) match {
						case None => Console.println("Skipping bad rectangle")
						case Some(region) =>
	
							//*
							val gridImg = imgRemapped.submat(region.tl.y.intValue, region.br.y.intValue, region.tl.x.intValue, region.br.x.intValue)
							val dstImg = dpTarget.submat(region.tl.y.intValue, region.br.y.intValue, region.tl.x.intValue,  region.br.x.intValue)
							val instr = brickPlacer.evaluate(gridImg, brickLib, BrickSynth.COLOR_TABLE.toSet[(Scalar,Scalar)])
	
							instr.foreach{
							case(proj,colors,brick) =>
							brick.project(dstImg, colors, Some(proj))
							}
							//Util.makeImageFrame(Util.matToImage(gridImg), f"src $i")
							//Util.makeImageFrame(Util.matToImage(dstImg), f"solved $i")
							//Util.makeImageFrame(Util.matToImage(dpTarget), f"solved $i (whole)")
							//*/

					}
					
			}
			Util.makeImageFrame(Util.matToImage(dpTarget), "solved")
			
			
		} else {
			Console.println("Could not achieve a valid gridded facade.")
			Console.println("\t=> Try adjusting the mesh resolution or deletion weights")
		}
		
		
		/*
		val imageTest = Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE)
		Console.println("Testing: " + image.size)
		val wd = new SVMDetector("trained-svms/1418999570258_window_.opencv_svm",usedHogs,"trained-svms/window-posExSizes.histogram")
		val detected = wd.detect(imageTest).map{case(rect, response) => 
			Console.println("Found " + rect + " with response " + response)
			rect}.toList
		val assignedClusters = Util.clusterRects(detected)
		val clusterReps = assignedClusters.map{
				cluster =>
				val avgR = cluster.foldLeft((0,0.0,0.0,0.0,0.0)){
					case((n,tX,tY, bX, bY), rect) =>
						val nextN = n + 1
						(nextN, (tX * n) / nextN + (rect.tl.x) / nextN, (tY * n) / nextN + (rect.tl.y) / nextN, (bX * n) / nextN + (rect.br.x) / nextN, (bY * n) / nextN + (rect.br.y) / nextN)
				}
				new Rect(new Point(avgR._2, avgR._3), new Point(avgR._4, avgR._5))
			}
		//Console.println()
		*/
		
		/*
		clusterReps.foreach{
			case(found) =>
				Core.rectangle(imageTest, found.tl, found.br, new Scalar(1.0))
		}
		Util.makeImageFrame(Util.matToImage(imageTest),"window detections")
		*/
		
		

		//LearnerFactory.learnAll(usedHogs)
	}
}