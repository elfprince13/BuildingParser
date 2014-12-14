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
import org.opencv.core.Point

class Translate(ofs:Point) extends Transform {
		def transform(input:Point):Point = {
			new Point(ofs.x + input.x, ofs.y + input.y)
		}
}

class Prim2DRects(val data:Mat, val masks:Mat, nameFormatter:(Size => String)) extends Primitive {
	def numDims():Int = 2
	def varsStart():Int = 2
	def numColorVars():Int = 2
	def project(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Unit = {
		val origin = new Point(0,0)
		val ofs = txform match {
			case Some(t) => t.transform(origin)
			case None => origin
		}
		
		((0 until varsStart).map(i => (null, i)) ++ (colorVars zip (varsStart until (varsStart + numColorVars)))).foreach{
			case(c, i) =>
				val (src, mask) = /*if(i == 0) {
					(data, Mat.zeros(masks.size, masks.`type`))
				} else */{
					val fullVar = new Mat(masks.size, masks.`type`, new Scalar(i))
					val varOnly = new Mat
					Core.compare(masks, fullVar, varOnly, Core.CMP_EQ)
					(if(i < varsStart){data}else{new Mat(data.size, data.`type`, c)}, varOnly)
				}
				
				src.copyTo(dst.submat(ofs.x.intValue, (ofs.x + bounds.width).intValue, ofs.y.intValue, (ofs.y + bounds.height).intValue), mask)
				
		}
	}
	
	def reward(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Double = {
		val origin = new Point(0,0)
		val ofs = txform match {
			case Some(t) => t.transform(origin)
			case None => origin
		}
		
		val primRepr = Mat.zeros(data.size, data.`type`)
		project(primRepr, colorVars, None)
		
		val dstChunk = new Mat(data.size, data.`type`)
		dst.submat(ofs.x.intValue, (ofs.x + bounds.width).intValue, ofs.y.intValue, (ofs.y + bounds.height).intValue).copyTo(dstChunk)
		
		val chunkSmooth = new Mat
		Imgproc.adaptiveBilateralFilter(dstChunk, chunkSmooth, new Size(5, 5), 2)
		
		val diff = new Mat
		Core.absdiff(chunkSmooth, primRepr, diff)
		
		
		0
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
