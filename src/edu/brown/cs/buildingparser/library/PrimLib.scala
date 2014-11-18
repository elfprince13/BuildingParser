package edu.brown.cs.buildingparser.library

import org.opencv.core.Point
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import java.io.File
import org.opencv.core.MatOfKeyPoint

trait Transform {
	def transform(input:Point):Point
}

trait Primitive {
	def numDims():Int
	def numColorVars():Int
	def project(dst:Mat, colorVars:List[Scalar], txform:Option[Transform]):Unit
	def bounds():Size
	def name():String
}

trait PrimLib {
	def loadFromDirs(srcs:List[File]):(List[Primitive],Map[String, List[Primitive]])
	def computeSIFTs(prims:List[Primitive]):Map[Primitive,(MatOfKeyPoint,Mat)]
}