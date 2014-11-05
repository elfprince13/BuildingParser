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

class ExtractSamples(imgDir:String, labelDir:String, destDir:String, exts:Set[String], 
		bounds:Size, srcLabelMap:Map[String,(Scalar,Boolean)], dstLabelMap:Map[(Scalar,Boolean),String]) {
	val BOUNDARY_WIDTH = 128
	
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
			assert(FilenameUtils.getBaseName(srcHandle.getName) == FilenameUtils.getBaseName(labelHandle.getName))
			val srcImg = Util.makeBoundaryMirrored(Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR), BOUNDARY_WIDTH)
			val labelImg = Highgui.imread(labelHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)
			
			
			assert(srcImg.rows == labelImg.rows && srcImg.cols == labelImg.cols)
			
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
						case(box,i) => if(box.height > bounds.height || box.width > bounds.width){
							throw new ExtractorException("This " + labelName + " is too big for this window")
						} else {
							val grabW = bounds.width.toInt
							val grabH = bounds.height.toInt
							val xdiff = grabW - box.width
							val ydiff = grabH - box.height
							val x = if(box.x + grabW - xdiff / 2 > srcImg.cols){ srcImg.cols - grabW } else { Math.max(0,box.x - xdiff / 2) }
							val y = if(box.y + grabH - ydiff / 2 > srcImg.rows){ srcImg.rows - grabH } else { Math.max(0,box.y - ydiff / 2) }
							val grabBox = new Rect(x,y,grabW,grabH)
							val grabMat = srcImg.submat(grabBox)
							val newName = destDir + File.separator + dstLabelMap((labelColor,true)) + File.separator + FilenameUtils.getBaseName(srcHandle.getName) + "_" + i + ".png" 
							val success = Highgui.imwrite(newName,grabMat)
						}
					}
				} else {
					;
				}
			}
			
	}

}

class ExtractorException(msg:String) extends Exception(msg);

