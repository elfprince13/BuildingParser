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
	def reward(against:Mat, colorVars:List[Scalar], txform:Option[Transform]):Double
	def bounds():Size
	def name():String
}

trait PrimLib[T <: Primitive] {
	def loadFromDirs(specialKinds:Set[String], srcs:List[File]):(List[T],Map[String, List[T]])
	def computeSIFTs(prims:List[T]):Map[T,(MatOfKeyPoint,Mat)]
}