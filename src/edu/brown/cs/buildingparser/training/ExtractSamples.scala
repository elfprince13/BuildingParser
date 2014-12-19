package edu.brown.cs.buildingparser.training

import org.opencv.core.Scalar
import org.opencv.core.Size
import org.apache.commons.io.FilenameUtils
import java.io.File
import scala.util.Sorting
import org.opencv.highgui.Highgui
import edu.brown.cs.buildingparser.Util
import org.opencv.core.Mat
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfPoint
import scala.collection.JavaConverters._
import org.opencv.core.Rect
import org.opencv.core.Range
import scala.util.Random
import edu.brown.cs.buildingparser.Util

class ExtractSamples(imgDir:String, labelDir:String, destDir:String, antiDestDir:String, exts:Set[String], 
		bounds:List[Size], srcLabelMap:Map[String,(Scalar,Boolean)], dstLabelMap:Map[(Scalar,Boolean),String]) {
	val BOUNDARY_WIDTH = 128
	val coordGen = new Random()
	val INV_OVERLAP_THRESHOLD = 3
	val COUNTER_EXAMPLES_PER_BINSIZE = 4
	val MAX_RANDOM_DEPTH = 20
	
	def bestBoundsBucket(box:Rect):Size = Util.bestBoundsBucket(box,bounds)
	
	def findCounterExample(srcImg: Mat, exclBoxes: Traversable[Rect], bound: Size, depth:Int = 0):Rect = {
		val x = coordGen.nextInt(srcImg.width - 2 * BOUNDARY_WIDTH - bound.width.intValue) + BOUNDARY_WIDTH
		val y = coordGen.nextInt(srcImg.height - 2 * BOUNDARY_WIDTH - bound.height.intValue) + BOUNDARY_WIDTH
		val counterRect = new Rect(x, y, bound.width.intValue, bound.height.intValue)
		val xInt = (new Range(counterRect.x, counterRect.x + counterRect.width))
		val yInt = (new Range(counterRect.y, counterRect.y + counterRect.width))
		if(exclBoxes.forall( box =>
			xInt.intersection(new Range(box.x, box.x + box.width)).size() *
			yInt.intersection(new Range(box.y, box.y + box.height)).size() *
			INV_OVERLAP_THRESHOLD < box.area()) ){
			counterRect
		} else {
			if(depth >= MAX_RANDOM_DEPTH){
				//Console.println("Counter example search-depth exceeded")
				null 
			} else{ findCounterExample(srcImg, exclBoxes, bound, depth+1) }
		}
	}
	
	def extractLabeledRects(labelImg:Mat, labelColor:Scalar, boundaryWidth:Int = 0):List[Rect] = {
		val fullLabelColor = new Mat(labelImg.rows,labelImg.cols,labelImg.`type`,labelColor)
		val labelOnly = new Mat
		Core.compare(labelImg,fullLabelColor,labelOnly,Core.CMP_EQ)
		val labelBW = new Mat
		Imgproc.cvtColor(labelOnly, labelBW, Imgproc.COLOR_BGR2GRAY)
		val labelMask = new Mat
		Imgproc.threshold(labelBW, labelMask, 254.9, 255, Imgproc.THRESH_BINARY)
		
		val contours = new java.util.LinkedList[MatOfPoint]
		val hierarchy = new Mat
		Imgproc.findContours(Util.makeBoundaryFilled(labelMask, boundaryWidth), contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
		contours.asScala.view.map(Imgproc.boundingRect).toList
	}
	
	def grabBucketed(srcImg:Mat, boxes:List[Rect]) = {
		boxes.view.zipWithIndex.map{
			case(box,i) =>
				val bestBounds = bestBoundsBucket(box)
				if(bestBounds == null){
					null
				} else {
					val grabBox = Util.calcGrabBox(box, bestBounds)
					val grabMat = srcImg.submat(grabBox)
					(box,i,grabMat)
				}				
		}
	}


	def stripExts(handle:File):File = {
			new File(FilenameUtils.getBaseName(handle.getName))
	}

	val imgDirHandle = new File(imgDir)
	val labelDirHandle = new File(labelDir)
	val srcImgs:Set[File] = Util.filterContentsByExts(imgDirHandle,exts)
	val labelImgs:Set[File] = Util.filterContentsByExts(labelDirHandle,exts)

	// Assumption: There is only one file with the same basename per directory
	// Also this makes a lot of garbage, but it should only run a few times
	val inpSubset = (srcImgs.map(stripExts) intersect labelImgs.map(stripExts)).flatMap(img => exts.map(ext => img.getName + ext))
	/*
	 inpSubset.toList.sorted.foreach{
		f => Console.println("Intersected: " + f)
	}
	 */
	val pairedImgs = { 
		val allowedSrcs = srcImgs.filter(f => inpSubset.contains(f.getName)).toList.sorted
		val allowedLabels = labelImgs.filter(f => inpSubset.contains(f.getName)).toList.sorted 
		(allowedSrcs zip allowedLabels)
	}.toMap
	
	def imgHandleFromName(name:String) = { new File (imgDir + File.separator + name) }
	def extractOneExampleSet(srcHandle:File, labelHandle:File) = {
		val srcBase = FilenameUtils.getBaseName(srcHandle.getName)
		val labelBase = FilenameUtils.getBaseName(labelHandle.getName)
		assert(srcBase == labelBase)
		val readImg = Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)
		val labelImg = Highgui.imread(labelHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)

		assert(readImg.rows == labelImg.rows && readImg.cols == labelImg.cols)
		val srcImg = Util.makeBoundaryMirrored(readImg, BOUNDARY_WIDTH)
		
		(srcBase, srcLabelMap.map{
			case(labelName,(labelColor,isObjectLabel)) => 
				if(isObjectLabel){
					val boxes = extractLabeledRects(labelImg, labelColor, BOUNDARY_WIDTH)
					val grabbedPos = grabBucketed(srcImg, boxes)
					val grabbedNeg = bounds.view.map{ bound => 
						grabBucketed(srcImg,Seq.fill(COUNTER_EXAMPLES_PER_BINSIZE)(findCounterExample(srcImg, boxes, bound)).filter(rect => rect != null).toList)
					}.flatten
					(labelName, labelColor, grabbedPos, grabbedNeg)
				} else {
					throw new ExtractorException("We don't sample background patches");
				}
			})
	}
	
	def extractAndSaveAll():Unit = {
		pairedImgs.foreach{
			case (srcHandle, labelHandle) =>
				val (srcBase, extractedHere) = extractOneExampleSet(srcHandle, labelHandle)
				extractedHere.foreach{
					case(labelName, labelColor, posStream, negStream) =>
						posStream.foreach{
							case(box, i, grabMat) =>
								if(grabMat == null){
									val errString = "This " + labelName + " is too big for any of our windows: " + box
									Console.println(errString)
									//throw new ExtractorException(errString)
								} else {
									val newName = destDir + File.separator + dstLabelMap((labelColor,true)) + File.separator + srcBase + "_" + i + ".png" 
									val success = Highgui.imwrite(newName,grabMat)
								}
						}
						
						negStream.foreach{
							case(box, i, grabMat) => 
								if(grabMat == null){
									;
									//Console.println("Warning: could find no " + labelName + "-free regions of size " + bound)
								} else{
									val dirPrefix = antiDestDir + File.separator + dstLabelMap((labelColor,true)) + File.separator + (box.width.intValue + "x" + box.height.intValue)
									val newName = dirPrefix + File.separator + srcBase + "_" + i + ".png"
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
		}
	}
	
}

class ExtractorException(msg:String) extends Exception(msg);

object SamplerMain {
	
	val stdLabelProps = Map[String,(Scalar,Boolean)](
						"window" -> (new Scalar(0,0,255),true),
						//"balcony" -> (new Scalar(255,0,128),true),
						"door" -> (new Scalar(0,128,255),true))
	
	val dimBuckets = List(16, 32, 48, 64, 96, 128)	
	val boundBuckets = (for( i <- dimBuckets; j <- dimBuckets) yield Pair(i,j)) map {
		case (i, j) => new Size(i, j)
	}
	
	var isMain = true
	
	def makeStandardSampler(imgDir:String, labelDir:String) = {
		new ExtractSamples(imgDir, labelDir,
				"object-classes", "anti-object-classes", Set[String](".png",".jpg",".jpeg"),
				boundBuckets, 
				stdLabelProps, 
				stdLabelProps.map(kvp => (kvp._2, kvp._1)))
	}
	
	def parisSampler() = {
		makeStandardSampler(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionAndGraphics/cvpr2010/images",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionAndGraphics/ground_truth_2011.zip Folder")
	}
	
	def grazSampler() = {
		makeStandardSampler(
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionAndGraphics/graz50_facade_dataset.zip Folder/graz50_facade_dataset/images",
				"/Users/thomas/Documents/Brown/CS2951B-DataDrivenVisionAndGraphics/graz50_facade_dataset.zip Folder/graz50_facade_dataset/labels_full")
	}
	
	def main(args:Array[String]):Unit = {
		if(isMain){
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		}
		val pSampler = parisSampler
		pSampler.extractAndSaveAll
		val gSampler = grazSampler
		gSampler.extractAndSaveAll
	}
}

