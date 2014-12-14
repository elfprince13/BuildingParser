package edu.brown.cs.buildingparser.library

import org.opencv.core.Size
import org.opencv.core.Rect
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File
import org.opencv.core.MatOfKeyPoint
import org.opencv.highgui.Highgui
import org.opencv.core.MatOfPoint
import org.opencv.imgproc.Imgproc
import org.opencv.core.Core
import collection.JavaConverters._
import org.opencv.features2d.FeatureDetector
import org.opencv.features2d.DescriptorExtractor

class Prim2DRects(val data:Mat, val masks:Mat, nameFormatter:(Size => String)) extends Primitive {
	def numDims():Int = 2
	def numColorVars():Int = 2
	def project(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Unit = {
		
	}
	def bounds():Size = data.size
	def name():String = nameFormatter(bounds)
}

object Prim2DRectsLib extends PrimLib[Prim2DRects] {
	def loadFromDirs(specialKinds:Set[String], srcs:List[File]) : (List[Prim2DRects],Map[String, List[Prim2DRects]]) = {
		val primImgs = srcs.map{
			f => 
				val fAbs = f.getAbsoluteFile
				val fDir = f.getParent
				val fName = f.getName
				(fName.split("-")(0),
						Highgui.imread(fDir + File.pathSeparator + "mask_" + fName, Highgui.CV_LOAD_IMAGE_GRAYSCALE),
			Highgui.imread(f.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR))
		}
		  
		val allPrims = primImgs.map{
			case(kind, varsMask, img) =>
				/*
				val transCol = Mat.zeros(varsMask.size, varsMask.`type`)
				val transMask = new Mat
				Core.compare(varsMask, transCol, transMask, Core.CMP_EQ)
				
				val contours = new java.util.LinkedList[MatOfPoint]
				val hierarchy = new Mat
				Imgproc.findContours(transMask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
				val holes = contours.asScala.view.map(Imgproc.boundingRect).toList
				*/
				(kind, new Prim2DRects(img, varsMask, ((sz:Size) => f"$kind - ${sz.width.intValue}%d x ${sz.height.intValue}%d")))
		}
		allPrims.foldLeft((List[Prim2DRects](),Map[String, List[Prim2DRects]]())){
			case(lastOut, in) =>
				val (std, special) = lastOut
				val (kind, prim) = in
				if (specialKinds.contains(kind)) {
					(std :+ prim, special)
				} else {
					val kindUpd = (special.getOrElse(kind, List[Prim2DRects]()) :+ prim)
					(std, special + (kind -> kindUpd))
				}
		}
	}
	
	def computeSIFTs(prims:List[Prim2DRects]):Map[Prim2DRects,(MatOfKeyPoint,Mat)] = {
		val siftDetector = FeatureDetector.create(FeatureDetector.SIFT)
		val siftExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT)
		prims.map{
			prim =>
				val grey = new Mat
				val kp = new MatOfKeyPoint
				val desc = new Mat		
				Imgproc.cvtColor(prim.data, grey, Imgproc.COLOR_BGR2GRAY)
				siftDetector.detect(grey, kp)
				siftExtractor.compute(grey, kp, desc)
				(prim -> (kp, desc))
		}.toMap
	}
}
