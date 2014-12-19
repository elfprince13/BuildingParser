package edu.brown.cs.buildingparser.library

import org.opencv.core.Size
import org.opencv.core.Rect
import org.opencv.core.Mat
import org.opencv.core.CvType
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

import edu.brown.cs.buildingparser.Util

class Translate(ofs:Point) extends Transform {
		def transform(input:Point):Point = {
			new Point(ofs.x + input.x, ofs.y + input.y)
		}
}

class Prim2DRect(val data:Mat, val masks:Mat, nameFormatter:(Size => String)) extends Primitive {
	def numDims():Int = 2
	def varsStart():Int = 2
	def numColorVars():Int = 2
	def project(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Unit = {
		val origin = new Point(0,0)
		val ofs = txform match {
			case Some(t) => t.transform(origin)
			case None => origin
		}
		//Console.println("Projecting to ${ofs.x} x ${ofs.y}")
		
		((0 until varsStart).map(i => (null, i)) ++ (colorVars zip (varsStart until (varsStart + numColorVars)))).foreach{
			case(c, i) =>
				val (src, mask) = /*if(i == 0) {
					(data, Mat.zeros(masks.size, masks.`type`))
				} else */{
					val fullVar = new Mat(masks.size, masks.`type`, new Scalar(i))
					val varOnly = new Mat
					Core.compare(masks, fullVar, varOnly, Core.CMP_EQ)
					//Util.makeImageFrame(Util.matToImage(varOnly), f"mask $i / $varsStart for ${name}")
					(if(i < varsStart){data}else{new Mat(data.size, data.`type`, c)}, varOnly)
				}
				
				src.copyTo(dst.submat(ofs.y.intValue, (ofs.y + bounds.height).intValue, ofs.x.intValue, (ofs.x + bounds.width).intValue), mask)
				
		}
	}
	
	def reward(dst:Mat, colorVars:List[Scalar], txform:Option[Transform], params:Map[String,Double] = Map("l" -> 0.25, "k" -> 1)):Double = {
		val origin = new Point(0,0)
		val ofs = txform match {
			case Some(t) => t.transform(origin)
			case None => origin
		}
		
		val primRepr = Mat.zeros(data.size, data.`type`)
		project(primRepr, colorVars, None)
		
		val dstChunk = new Mat(data.size, data.`type`)
		dst.submat(ofs.y.intValue, (ofs.y + bounds.height).intValue, ofs.x.intValue, (ofs.x + bounds.width).intValue).copyTo(dstChunk)
		
		//val chunkSmooth = new Mat
		//Imgproc.adaptiveBilateralFilter(dstChunk, chunkSmooth, new Size(5, 5), 2)
		
		val dstHSV = new Mat
		Imgproc.cvtColor(dstChunk, dstHSV, Imgproc.COLOR_BGR2HSV)
		val primHSV = new Mat
		Imgproc.cvtColor(primRepr, primHSV, Imgproc.COLOR_BGR2HSV)
		
		val diff = new Mat
		Core.absdiff(dstHSV, primHSV, diff)
		
		
		(0 until diff.height).map{ 
			r => 
				(0 until diff.width).map{
					c =>
						val cDiffs = diff.get(r, c)
						val dLen = Math.sqrt(cDiffs.foldLeft(0.0)((b,a) => b + a*a))
						val okp = 1.0 + params("k")*dLen
						1.0 / Math.pow(okp,1)
				}
		}.flatten.foldLeft(0.0)(_+_) / Math.pow(diff.size.area, 1 - params("l"))
	}
	
	def bounds():Size = data.size
	def name():String = nameFormatter(bounds)
}

object Prim2DRectsLib extends PrimLib[Prim2DRect] {
	def formatterForKind(kind:String):(Size => String) = {
		((sz:Size) => f"$kind - ${sz.width.intValue}%d x ${sz.height.intValue}%d")
	}
	def loadFromDirs(specialKinds:Set[String], srcs:List[File]) : (List[Prim2DRect],Map[String, List[Prim2DRect]]) = {
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
				(kind, new Prim2DRect(img, varsMask, formatterForKind(kind)))
		}
		allPrims.foldLeft((List[Prim2DRect](),Map[String, List[Prim2DRect]]())){
			case(lastOut, in) =>
				val (std, special) = lastOut
				val (kind, prim) = in
				if (specialKinds.contains(kind)) {
					(std :+ prim, special)
				} else {
					val kindUpd = (special.getOrElse(kind, List[Prim2DRect]()) :+ prim)
					(std, special + (kind -> kindUpd))
				}
		}
	}
	
	def computeSIFTs(prims:List[Prim2DRect]):Map[Prim2DRect,(MatOfKeyPoint,Mat)] = {
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

object BrickSynth {
	def bricksFromDims(dims:List[Size], gridStep:(Int,Int)):List[Prim2DRect] = {
		dims.map{
			sz =>
				val pxWidth = (sz.width * gridStep._1).intValue
				val pxHeight = (sz.height * gridStep._2).intValue
				//Console.println(f"Making $pxWidth x $pxHeight (rows: $pxHeight, columns: $pxWidth)")
				val data = Mat.zeros(pxHeight, pxWidth, CvType.CV_8UC3)
				val masks = new Mat(pxHeight, pxWidth, CvType.CV_8UC1, new Scalar(2))
				Core.rectangle(masks, new Point(0,0), new Point(pxWidth - 1, pxHeight - 1), new Scalar(3), 1)
				new Prim2DRect(data, masks, Prim2DRectsLib.formatterForKind("brick"))
		}
	}
	
	val PLATE_HEIGHT = 1
	val BRICK_HEIGHT = 3
	val BRICK_STEP = (20,8)
	
	val STD_DIMS = List((List(1, 2, 3, 4, 6, 8, 10, 12, 16), List(PLATE_HEIGHT, BRICK_HEIGHT)),(List(1,2),List(2*BRICK_HEIGHT, 5*BRICK_HEIGHT)))
	
	val COLOR_TABLE = List((new Scalar(0x1D, 0x13, 0x05), new Scalar(0x59, 0x59, 0x59)),
			(new Scalar(0xBF, 0x55, 0x00), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x3E, 0x7A, 0x25), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x8F, 0x83, 0x00), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x09, 0x1A, 0xC9), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xA0, 0x70, 0xC8), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x27, 0x39, 0x58), new Scalar(0x1E, 0x1E, 0x1E)),
			(new Scalar(0x9D, 0xA1, 0x9B), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x5C, 0x6E, 0x6D), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xE3, 0xD2, 0xB4), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x4A, 0x9F, 0x4B), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xAF, 0xA5, 0x55), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x5E, 0x70, 0xF2), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xAC, 0x97, 0xFC), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x37, 0xCD, 0xF2), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xFF, 0xFF, 0xFF), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xB8, 0xDA, 0xC2), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x96, 0xE6, 0xFB), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x9E, 0xCD, 0xE4), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xE2, 0xCA, 0xC9), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x7B, 0x00, 0x81), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xB0, 0x32, 0x20), new Scalar(0x1E, 0x1E, 0x1E)),
			(new Scalar(0x18, 0x8A, 0xFE), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x78, 0x39, 0x92), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x0B, 0xE9, 0xBB), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x73, 0x8A, 0x95), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xC8, 0xAD, 0xE4), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xBA, 0x78, 0xAC), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xED, 0xD5, 0xE1), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x9B, 0xCF, 0xF3), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x98, 0x62, 0xCD), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x12, 0x2A, 0x58), new Scalar(0x59, 0x59, 0x59)),
			(new Scalar(0xA9, 0xA5, 0xA0), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x68, 0x6E, 0x6C), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xD1, 0x9D, 0x5C), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xA1, 0xDC, 0x73), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xCF, 0xCC, 0xFE), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xB3, 0xD7, 0xF6), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x2A, 0x70, 0xCC), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x91, 0x36, 0x3F), new Scalar(0x1E, 0x1E, 0x1E)),
			(new Scalar(0x3A, 0x50, 0x7C), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0xDB, 0x61, 0x4C), new Scalar(0x33, 0x33, 0x33)),
			(new Scalar(0x68, 0x91, 0xD0), new Scalar(0x33, 0x33, 0x33)))
	
	def getStdBricks():List[Prim2DRect] = {
		STD_DIMS.map{
			case(widths, heights) =>
				bricksFromDims(widths.map{
					width =>
						heights.map{
							height =>
								new Size(width, height)
						}
				}.flatten, BRICK_STEP)
		}.flatten
	}
}

case class WindowSplit(vert:Boolean, frac:Double, children:Option[(WindowSplit,WindowSplit)])

object WindowSynth {
	val PLATE_HEIGHT = BrickSynth.PLATE_HEIGHT
	val BRICK_HEIGHT = BrickSynth.BRICK_HEIGHT
	val BRICK_STEP = BrickSynth.BRICK_STEP
	val COLOR_TABLE = BrickSynth.COLOR_TABLE
	def drawPanes(dst:Mat,rect:Rect, panes:WindowSplit) = {
		panes match {
			case WindowSplit(vert, frac, children) =>
				children match {
					case None =>
						Core.rectangle(dst, rect.tl, rect.br, new Scalar(3))
					case Some(children) =>
						
				}
				
		}
	}
}

object BrickTest {
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		val primset = BrickSynth.getStdBricks()
		val C_DIVISOR = 8
		primset.foreach{
			prim =>
				val dst = new Mat(prim.data.height * Math.ceil(BrickSynth.COLOR_TABLE.length / C_DIVISOR.floatValue).intValue, prim.data.width * C_DIVISOR, CvType.CV_8UC3)
				Console.println(f"dst for ${prim.name} is ${prim.data.width * C_DIVISOR} x ${prim.data.height * (BrickSynth.COLOR_TABLE.length / C_DIVISOR)}")
				BrickSynth.COLOR_TABLE.zipWithIndex.foreach{
					case((face_color, edge_color),i) =>
						Console.println(f"Translate $i to ${(i % C_DIVISOR) * prim.data.width} x ${(i / C_DIVISOR) * prim.data.height}")
						prim.project(dst, List(face_color, edge_color), Some(new Translate(new Point((i % C_DIVISOR) * prim.data.width,(i / C_DIVISOR) * prim.data.height))))
				Console.println("test")
				}
				Util.makeImageFrame(Util.matToImage(dst), prim.name)
		}
	}
}