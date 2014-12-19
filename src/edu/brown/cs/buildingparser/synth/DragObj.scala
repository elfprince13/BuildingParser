package edu.brown.cs.buildingparser.synth
import org.opencv.core.Mat
import org.opencv.core.Rect


// Would really like to do this with a UV based dragged, or something else
trait DragObj {
	def dragObjs(src:Mat, dst:Mat, objs:List[(Rect,Rect)]):Unit
}