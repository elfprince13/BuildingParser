package edu.brown.cs.buildingparser.synth
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size


// Would really like to do this with a UV based dragged, or something else
abstract class DragObj(srcSz:Size, dstSz:Size, objs:List[(Rect,Rect)]) {
	def dragObjs(src:Mat, dst:Mat):Unit
}