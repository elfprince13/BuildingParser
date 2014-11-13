package edu.brown.cs.buildingparser.synth

import org.opencv.core.Size
import org.opencv.core.Point
import org.opencv.core.Rect
import edu.brown.cs.buildingparser.Util

import oscar.cp.modeling._
import oscar.cp.core._

class ObjConstraints(boundary:Size, objs:Map[String,Map[Int,List[Rect]]], gridStep:(Int, Int)) extends CPModel {
	def zpInt() = { CPIntVar(0 to Int.MaxValue)}
	def snapX(v:CPIntVar) = snapConstraint(v, gridStep._1)
	def snapY(v:CPIntVar) = snapConstraint(v, gridStep._2)
	def snapConstraint(v:CPIntVar, step:Int) = { (v % step == 0) }
	val boundaryTargets:Map[String,Int] = Map(
			"xMin" -> 0, 
			"yMin" -> 0, 
			"xMax" -> boundary.width.intValue, 
			"yMax" -> boundary.height.intValue)
	val boundaryVars:Map[String,CPIntVar] = Map(
			"xMax" -> zpInt,
			"yMax" -> zpInt)
	
	val objTargets:Map[String,Map[Int,List[Map[String,Int]]]] = objs.map{
		case(labelName,clusters) => (labelName -> clusters.map{ 
			case(clusterNumber, boxes) => (clusterNumber -> boxes.map{
				box => Map(
					"xMin" -> box.tl.x.intValue,
					"yMin" -> box.tl.y.intValue,
					"xMax" -> box.br.x.intValue,
					"yMax" -> box.br.y.intValue)})})}
	
	val objVars = objTargets.map{
		case(labelName,clusters) => (labelName -> clusters.map{ 
			case(clusterNumber, boxes) => (clusterNumber -> 
			boxes.map( box => box.map{case(varName,_) => (varName -> zpInt)}))})}
	
	def objStream() = {
		objVars.view.map{
			case(labelName,clusters) => clusters.view.map{ 
				case(clusterNumber, boxes) => 
					boxes.view}.flatten }.flatten
	}
	
	def boxesNotInsideOut() = {
		objStream.foreach{
			box =>
				add(box("xMin") <= box("xMax"))
				add(box("yMin") <= box("yMax"))
		}
	}
	
	def boxesInsideBoundary() = {
		objStream.foreach{
			box =>
				add(box("xMin") >= boundaryTargets("xMin"))
				add(box("yMin") >= boundaryTargets("yMin"))
				add(box("xMax") <= boundaryVars("xMax"))
				add(box("yMax") <= boundaryVars("yMax"))
		}
	}
	
	def noIntersections() = {
		objStream.toStream.zipWithIndex.foreach{
			case(oBox,i) =>
				objStream.take(i - 1).foreach{
					case(iBox) =>
						val xIntersection = (
								(oBox("xMin") <== iBox("xMin")) && 
								(iBox("xMin") <<= oBox("xMax"))) || 
								((iBox("xMin") <== oBox("xMin")) && 
										(oBox("xMin") <<= iBox("xMax")))
						val yIntersection = (
								(oBox("yMin") <== iBox("yMin")) && 
								(iBox("yMin") <<= oBox("yMax"))) || 
								((iBox("yMin") <== oBox("yMin")) && 
										(oBox("yMin") <<= iBox("yMax")))
						add(!(xIntersection) || !(yIntersection) )
				}
		}
	}
	
	def snappedToGrid() = {
		add(snapX(boundaryVars("xMax")))
		add(snapY(boundaryVars("yMax")))
		objStream.foreach{
			case(box) =>
				add(snapX(box("xMin")))
				add(snapY(box("yMin")))
				add(snapX(box("xMax")))
				add(snapY(box("yMax")))
		}
	}
	
	def addAllConstraints() = {
		boxesNotInsideOut()
		boxesInsideBoundary()
		noIntersections()
		snappedToGrid()
	}
	
}

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