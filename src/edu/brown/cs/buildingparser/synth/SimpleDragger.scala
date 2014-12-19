package edu.brown.cs.buildingparser.synth

import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Rect
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc

import jsat.linear.vectorcollection.KDTree
import jsat.linear.DenseVector
import jsat.linear.VecPaired
import jsat.linear.distancemetrics.EuclideanDistance

import scala.collection.JavaConverters._

class SimpleDragger extends DragObj{
	def pt2Vec(pt:Point):DenseVector = {
		new DenseVector(List(pt.x, pt.y).map(_.asInstanceOf[java.lang.Double]).asJava)
	}
	def vec2Pt(vec:DenseVector):Point = {
		new Point(vec.get(0), vec.get(1))
	}
	def dragObjs(src:Mat, dst:Mat, objs:List[(Rect,Rect)]):Unit = {
		val srcBound = new Rect(new Point(0,0), src.size)
		val dstBound = new Rect(new Point(0,0), dst.size)
		val keypointMap = (objs :+ (dstBound, srcBound)).map{
			case(oldR, newR) =>
				List(
						(oldR.tl, newR.tl),
						(oldR.br, newR.br),
						(new Point(oldR.tl.x, oldR.br.y), new Point(newR.tl.x, newR.br.y)),
						(new Point(oldR.br.x, oldR.tl.y), new Point(newR.br.x, newR.tl.y)))
		}.flatten.toMap
		
		val keypointSrcs = keypointMap.keySet.view.map(pt2Vec).toList.asJava
		
		val knn = new KDTree[DenseVector](keypointSrcs, new EuclideanDistance())
		
		val mapX = new Mat(dst.size, CvType.CV_32FC1)
		val mapY = new Mat(dst.size, CvType.CV_32FC1)
		
		(0 until dst.rows).foreach{
			r => (0 until dst.cols).foreach{
				c => 
					val here = new Point(c, r)
					val hereVec = pt2Vec(here)
					val ourKPs = objs.view.map(_._1).foldLeft(None.asInstanceOf[Option[Rect]]){
						case(found, testRect) =>
							if(testRect.contains(here)) {
								found match {
									case Some(fR) => throw new IllegalStateException("Objs overlap, this won't work")
									case None => Some(testRect)
								}
							} else {
								found
							}
					} match {
						case None => 
							knn.search(hereVec, 3).asScala
						case Some(parentRect) => 
							Seq(new VecPaired[DenseVector,java.lang.Double](pt2Vec(parentRect.tl),1))
					}
					
					val ourDelta = ourKPs.foldLeft((0.0,0.0)){
						case((dx, dy),kpPair) =>
							val kpP = vec2Pt(kpPair.getVector)
							val kpD = kpPair.getPair
							val kpDst = keypointMap(kpP)
							(dx + Math.abs((kpDst.x - kpP.x) / (1 + Math.abs(kpD))), dy + Math.abs((kpDst.y - kpP.y) / (1 + Math.abs(kpD))))
					}
					val ourNorm = ourKPs.foldLeft(0.0){
						case(a, kpPair) => a + (1 / (1 + Math.abs(kpPair.getPair)))
					}
					mapX.put(r, c, Array.fill(1)(here.x.floatValue + (ourDelta._1 / ourNorm).floatValue))
					mapY.put(r, c, Array.fill(1)(here.y.floatValue + (ourDelta._1 / ourNorm).floatValue))
					
			}
		}
		val fastMapPairs = new Mat
		val fastMapInterp = new Mat
		Imgproc.convertMaps(mapX, mapY, fastMapPairs, fastMapInterp, CvType.CV_16SC2)
		Imgproc.remap(src, dst, fastMapPairs, fastMapInterp, Imgproc.INTER_LINEAR, Imgproc.BORDER_REPLICATE, new Scalar(0))
	}
}