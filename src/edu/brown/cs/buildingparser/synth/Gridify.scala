package edu.brown.cs.buildingparser.synth

import org.opencv.core.Size
import org.opencv.core.Point
import org.opencv.core.Rect
import edu.brown.cs.buildingparser.Util

/***********
 * 
 * A standard minifig-door is:
 * * 3 brick-widths wide (3*20 = 60LDU)
 * * less than 5 brick-heights high (5*24 = 120LDU)
 * * aspect ratio 1:2
 * A standard house door is:
 * * 36 inches wide
 * * 80 inches high
 * * aspect ratio 1:2.2
 * So peg 1 LDU = 1.6667"
 * and assume 32 pixels is 36 inches
 * So 1 LDU = 1.875 pixels
 * 
 ***********/

object LDrawGridify extends Gridify((15,8), (20,8))


class Gridify(pixel2GridRatio:(Int,Int), gridStep:(Int, Int)) {
	
	def pixelsToGridUnits(pixelCount:Int):Int = {
		val fracGrids = ((pixelCount.doubleValue * pixel2GridRatio._2) / pixel2GridRatio._1)
		fracGrids.round.intValue
	}
	
	def gridUnitsToPixels(gridCount:Int):Int = {
		val fracPixels = ((gridCount.doubleValue * pixel2GridRatio._1) / pixel2GridRatio._2)
		fracPixels.round.intValue
	}
	
	def nearestMeshPoint(gridCount:Int, gridStep:Int):Int = {
		val fracGrids = gridCount.doubleValue / gridStep
		fracGrids.round.intValue * gridStep
	}
	
	def snapToGrid(pxVal:Int, gridStep:Int):Int = {
		gridUnitsToPixels(nearestMeshPoint(pixelsToGridUnits(pxVal), gridStep))
	}
	
	def snapRectToGrid(inRect:Rect):Rect = {
		// Explicitly force all rects of the same size to scale to the same size!
		// Might get rounding errors with the 2-pt constructor
		new Rect(snapPointToGrid(inRect.tl), snapSizeToGrid(inRect.size))
	}
	
	def snapPointToGrid(inPoint:Point):Point = {
		new Point(snapToGrid(inPoint.x.intValue, gridStep._1),snapToGrid(inPoint.y.intValue, gridStep._2))
	}
	
	def snapSizeToGrid(inSize:Size):Size = {
		new Size(snapToGrid(inSize.width.intValue, gridStep._1),snapToGrid(inSize.height.intValue, gridStep._2))
	}
	
	def fixIntersections(objs:List[Rect]) = {
		Util.findAllIntersections(objs)
	}
	
}