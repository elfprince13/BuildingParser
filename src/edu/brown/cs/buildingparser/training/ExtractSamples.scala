package edu.brown.cs.buildingparser.training

import org.opencv.core.Scalar
import org.opencv.core.Size
import org.apache.commons.io.FilenameUtils
import java.io.File
import scala.util.Sorting
import org.opencv.highgui.Highgui
import edu.brown.cs.buildingparser.ui.Util
import org.opencv.core.Mat
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfPoint
import scala.collection.JavaConverters._
import org.opencv.core.Rect
import org.opencv.core.Range
import scala.util.Random

class ExtractSamples(imgDir:String, labelDir:String, destDir:String, antiDestDir:String, exts:Set[String], 
		bounds:List[Size], srcLabelMap:Map[String,(Scalar,Boolean)], dstLabelMap:Map[(Scalar,Boolean),String]) {
	val BOUNDARY_WIDTH = 128
	val coordGen = new Random()
	val INV_OVERLAP_THRESHOLD = 3
	val COUNTER_EXAMPLES_PER_BINSIZE = 4
	val MAX_RANDOM_DEPTH = 20

	def bestBoundsBucket(box:Rect):Size = {
			bounds.filter(bound => 
			bound.width >= box.width && 
			bound.height >= box.height).foldLeft(null.asInstanceOf[Size])(
					(best, proposed) => 
					if(best == null || Math.max(best.width - box.width, best.height - box.height) > Math.max(proposed.width - box.width, proposed.height - box.height)){
						proposed	
					} else {
						best
					})
	}
	
	def findCounterExample(srcImg: Mat, exclBoxes: Traversable[Rect], bound: Size, depth:Int = 0):Mat = {
		val x = coordGen.nextInt(srcImg.width - 2 * BOUNDARY_WIDTH - bound.width.intValue) + BOUNDARY_WIDTH
		val y = coordGen.nextInt(srcImg.height - 2 * BOUNDARY_WIDTH - bound.height.intValue) + BOUNDARY_WIDTH
		val counterRect = new Rect(x, y, bound.width.intValue, bound.height.intValue)
		val xInt = (new Range(counterRect.x, counterRect.x + counterRect.width))
		val yInt = (new Range(counterRect.y, counterRect.y + counterRect.width))
		if(exclBoxes.forall( box =>
			xInt.intersection(new Range(box.x, box.x + box.width)).size() *
			yInt.intersection(new Range(box.y, box.y + box.height)).size() *
			INV_OVERLAP_THRESHOLD < box.area()) ){
			srcImg.submat(counterRect)
		} else {
			if(depth >= MAX_RANDOM_DEPTH){ null }
			else{ findCounterExample(srcImg, exclBoxes, bound, depth+1) }
		}
	}

	def filterContentsByExts(handle:File):Set[File] = {
			(Set[File]() ++ handle.listFiles).filter(img => exts.exists( ext => img.getName.endsWith(ext) ) )
	}

	def stripExts(handle:File):File = {
			new File(FilenameUtils.getBaseName(handle.getName))
	}

	val imgDirHandle = new File(imgDir)
	val labelDirHandle = new File(labelDir)
	val srcImgs:Set[File] = filterContentsByExts(imgDirHandle)
	val labelImgs:Set[File] = filterContentsByExts(labelDirHandle)

	// Assumption: There is only one file with the same basename per directory
	// Also this makes a lot of garbage, but it should only run a few times
	val inpSubset = (srcImgs.map(stripExts) intersect labelImgs.map(stripExts)).flatMap(img => exts.map(ext => img.getName + ext))
	/*
	 inpSubset.toList.sorted.foreach{
		f => Console.println("Intersected: " + f)
	}
	 */

	val pairedImgs = (srcImgs.filter(
			f => inpSubset.contains(f.getName)
			).toList.sorted zip
			labelImgs.filter(
					f => inpSubset.contains(f.getName)
					).toList.sorted)


					pairedImgs.foreach{
					case (srcHandle, labelHandle) =>
						val srcBase = FilenameUtils.getBaseName(srcHandle.getName)
						val labelBase = FilenameUtils.getBaseName(labelHandle.getName)
						assert(srcBase == labelBase)
						val readImg = Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)
						val labelImg = Highgui.imread(labelHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)


						assert(readImg.rows == labelImg.rows && readImg.cols == labelImg.cols)
						val srcImg = Util.makeBoundaryMirrored(readImg, BOUNDARY_WIDTH)
					

						// Either an object label or a region label
						srcLabelMap.foreach{
						case(labelName,(labelColor,isObjectLabel)) => if(isObjectLabel){
							val fullLabelColor = new Mat(labelImg.rows,labelImg.cols,labelImg.`type`,labelColor)
							val labelOnly = new Mat
							Core.compare(labelImg,fullLabelColor,labelOnly,Core.CMP_EQ)
							val labelBW = new Mat
							Imgproc.cvtColor(labelOnly, labelBW, Imgproc.COLOR_BGR2GRAY)
							val labelMask = new Mat
							Imgproc.threshold(labelBW, labelMask, 254.9, 255, Imgproc.THRESH_BINARY)
							/*Console.println(labelImg.dump)
						Console.println(fullLabelColor.dump)
						Console.println(labelOnly.dump)
						Console.println(labelBW.dump)*/
							/*
						Util.makeImageFrame(Util.matToImage(labelOnly))
						Util.makeImageFrame(Util.matToImage(labelBW))
						Util.makeImageFrame(Util.matToImage(labelMask))
						Util.makeImageFrame(Util.matToImage(labelImg))
							 */

						val contours = new java.util.LinkedList[MatOfPoint]
						val hierarchy = new Mat
						Imgproc.findContours(Util.makeBoundaryFilled(labelMask, BOUNDARY_WIDTH), contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
						//Imgproc.drawContours(srcImg, contours, -1, labelColor)
						//Util.makeImageFrame(Util.matToImage(srcImg))

						val boxes = contours.asScala.map(Imgproc.boundingRect)
						boxes.zipWithIndex.foreach{
							case(box,i) =>
								val bestBounds = bestBoundsBucket(box)
								if(bestBounds == null){
									val errString = "This " + labelName + " is too big for any of our windows: " + box
									Console.println(errString)
									//throw new ExtractorException(errString)
								} else {
									val grabW = bestBounds.width.toInt
									val grabH = bestBounds.height.toInt
									val xdiff = grabW - box.width
									val ydiff = grabH - box.height
									val x = box.x - xdiff / 2
									val y = box.y - ydiff / 2
									val grabBox = new Rect(x,y,grabW,grabH)
									val grabMat = srcImg.submat(grabBox)
									val newName = destDir + File.separator + dstLabelMap((labelColor,true)) + File.separator + srcBase + "_" + i + ".png" 
									val success = Highgui.imwrite(newName,grabMat)
								}
								
						}
						for(bound <- bounds){
							val dirPrefix = antiDestDir + File.separator + dstLabelMap((labelColor,true)) + File.separator + (bound.width.intValue + "x" + bound.height.intValue)
							for (n <- 0 until COUNTER_EXAMPLES_PER_BINSIZE) {
								val grabMat = findCounterExample(srcImg, boxes, bound)
								if(grabMat == null){
									;
									//Console.println("Warning: could find no " + labelName + "-free regions of size " + bound)
								} else{
									val newName = dirPrefix + File.separator + srcBase + "_" + n + ".png"
										val success = Highgui.imwrite(newName, grabMat)
									if(!success){
										if((new File(dirPrefix)).mkdirs){
											Highgui.imwrite(newName,grabMat)
										} else {
											throw new ExtractorException("Couldn't create directories for: " + dirPrefix)
										}
									}
								}
							}
						}
					} else {
						throw new ExtractorException("We don't sample background patches");
					}
					}

	}

}

class ExtractorException(msg:String) extends Exception(msg);

