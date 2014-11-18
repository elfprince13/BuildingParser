package edu.brown.cs.buildingparser.library

import org.opencv.core.Size
import org.opencv.core.Rect
import org.opencv.core.Mat
import org.opencv.core.Scalar

class Prim2DRects(bounds:Size, holes:List[Rect], nameFormatter:(Size => String)) extends Primitive {
	def numDims():Int = 2
	def numColorVars():Int = 2
	def project(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Unit = {
		
	}
	def bounds():Size = bounds
	def name():String = nameFormatter(bounds)
}

object Prim2DRectsLib extends PrimLib {
	def loadFromDirs(srcs:List[File]):(List[Primitive],Map[String, List[Primitive]]) = {
		
	}
	
	def computeSIFTs(prims:List[Primitive]):Map[Primitive,(MatOfKeyPoint,Mat)] = {
		
	}
}