package edu.brown.cs.buildingparser.synth

import org.opencv.core.Size
import org.opencv.core.Point
import org.opencv.core.Rect
import edu.brown.cs.buildingparser.Util
import oscar.cp.modeling._
import oscar.cp.core._
import scala.util.Random
import oscar.cp.constraints.GrEq
import oscar.cp.constraints.LeEq

class ObjConstraints(boundary:Size, objs:Map[String,Map[Int,List[Rect]]], gridStep:(Int, Int)) extends CPModel {
	var solved = false
	val dims = List("x","y")
	final val globalMaxVal = (2 * Math.max(boundary.width, boundary.height)).intValue
	def zpIntX(maxN:Int = globalMaxVal) = zpInt(maxN, gridStep._1)
	def zpIntY(maxN:Int = globalMaxVal) = zpInt(maxN, gridStep._2)
	def zpInt(maxN:Int = globalMaxVal, byN:Int = 1) = { 
		val drawnFrom = (0 until maxN by byN).toSet
		Console.println("Drawing from: " + drawnFrom)
		CPIntVar(drawnFrom )
	}
	
	def snapByName(v:Int, name:String) = {
		if(name(0) == 'x'){
			snapX(v)
		} else if (name(0) == 'y') {
			snapY(v)
		} else {
			throw new IllegalStateException("unknown variable dim")
		}
	}
	def snapX(v:Int) = snapVal(v, gridStep._1)
	def snapY(v:Int) = snapVal(v, gridStep._2)
	def snapVal(v:Int, step:Int) = { 
		val out = ((v.doubleValue / step).round * step).intValue
		Console.println(v + " snapped to " + out + " (by " + step + ")")
		out
	}
	
	def sideLen(box:Map[String,CPIntVar],dim:String):CPIntVar = {
		box(dim+"Max") - box(dim+"Min")
	}
	
	Console.println("In-Window: " + boundary)
	val boundaryTargets:Map[String,Int] = Map(
			"xMin" -> 0, 
			"yMin" -> 0, 
			"xMax" -> boundary.width.intValue, 
			"yMax" -> boundary.height.intValue)
	val boundaryVars:Map[String,CPIntVar] = Map(
			"xMax" -> zpIntX(globalMaxVal),
			"yMax" -> zpIntY(globalMaxVal))
	
	val objDTargets:Map[String,Map[Int,Map[String,Int]]] = objs.map{
		case(labelName,clusters) => (labelName -> clusters.map{
			case(clusterNumber, boxes) =>
				val fBox = boxes(0)
				val clusterData = Map(
						"xSz" -> fBox.width.intValue,
						"ySz" -> fBox.height.intValue
						)
				(clusterNumber -> clusterData)
		})
	}
	
	val objTargets:Map[String,Map[Int,List[Map[String,Int]]]] = objs.map{
		case(labelName,clusters) => (labelName -> clusters.map{ 
			case(clusterNumber, boxes) => 
				Console.println(labelName + " " + clusterNumber)
				(clusterNumber -> boxes.map{
					box =>  
						Console.println("\t" + box)
						Map("xMin" -> box.tl.x.intValue,
							"yMin" -> box.tl.y.intValue)})})}
	
	val objDVars = objDTargets.map{
		case(labelName,clusters) => (labelName -> clusters.map{
			case(clusterNumber, box) => (clusterNumber -> 
			box.map{ case(varName, targ) =>
				val dim = varName(0)
				val varV = if(dim == 'x'){ zpIntX(2 * targ) } else if(dim == 'y') { zpIntY(2 * targ) } else { throw new IllegalStateException("unknown dim value")}
				(varName -> varV)})})}
	
	val objVars = objTargets.map{
		case(labelName,clusters) => (labelName -> clusters.map{ 
			case(clusterNumber, boxes) => (clusterNumber -> 
			boxes.map( box => box.map{case(varName,targ) => 
				val dim = varName(0)
				val varV = if(dim == 'x'){ zpIntX(2 * targ) } else if(dim == 'y') { zpIntY(2 * targ) } else { throw new IllegalStateException("unknown dim value")}
				(varName -> varV)}))})}
	
	def objVTStream() = {
		objVars.view.map{
			case(labelName,labelVars) =>
				val labelTargets = objTargets(labelName)
				val dLTs = objDTargets(labelName)
				val dLVs = objDVars(labelName)
				labelVars.view.map{
					case(clustNum, clusterVars) =>
						val clusterTargets = labelTargets(clustNum)
						val dCTs = dLTs(clustNum)
						val dCVs = dLVs(clustNum)
						(clusterVars.toStream zip clusterTargets.toStream).view.map{
							case(boxVars, boxTargets) => 
								val outPair = (boxVars ++ dCVs, boxTargets ++ dCTs)
								outPair
						}
				}.flatten
		}.flatten
	}
	
	def objStream() = {
		objVars.view.map{
			case(labelName,clusters) => 
				val dLVs = objDVars(labelName)
				clusters.view.map{
					case(clusterNumber, boxes) =>
						val dCVs = dLVs(clusterNumber)
						boxes.view.map(_ ++ dCVs)}.flatten }.flatten
	}
	
	def boxesInsideBoundary():List[Constraint] = {
		dims.view.map{
			dim =>
				// attempt to abstract over the two types of comparisons here went badly
				// because of the typing differences
				val minName = dim + "Min"
				val minVs = objStream.view.map( box => box(minName)).toArray
				val minVIdxes = CPIntVar(0 until minVs.length)
				val minC = minVs(minVIdxes) >= boundaryTargets(minName)
				
				val szName = dim + "Sz"
				val maxName = dim + "Max"
				val maxVs = objStream.view.map( box => box(szName) + box(minName)).toArray
				val maxVIdxes = CPIntVar(0 until maxVs.length)
				val maxC = maxVs(maxVIdxes) < boundaryTargets(maxName)
			Stream(minC, maxC)	
		}.flatten.toList
	}
	
	def noIntersections():List[Constraint] = {
		objStream.toStream.zipWithIndex.map{
			case(oBox,i) =>
				objStream.view.take(i - 1).map{
					case(iBox) =>
						val oBxMax = oBox("xMin") + oBox("xSz")
						val oByMax = oBox("yMin") + oBox("ySz")
						val iBxMax = iBox("xMin") + iBox("xSz")
						val iByMax = iBox("yMin") + iBox("ySz")
						val xIntersection = (
								(oBox("xMin") <== iBox("xMin")) && 
								(iBox("xMin") <<= oBxMax)) || 
								((iBox("xMin") <== oBox("xMin")) && 
										(oBox("xMin") <<= iBxMax))
						val yIntersection = (
								(oBox("yMin") <== iBox("yMin")) && 
								(iBox("yMin") <<= oByMax)) || 
								((iBox("yMin") <== oBox("yMin")) && 
										(oBox("yMin") <<= iByMax))
						val noIntersection = (!(xIntersection) || !(yIntersection) )
						noIntersection.constraintTrue
				}
		}.flatten.toList
	}
	
	def boundaryPositive():List[Constraint] = {
		List(	(boundaryVars("xMax") > boundaryTargets("xMin")),
				(boundaryVars("yMax") > boundaryTargets("yMin")))
	}
	
	private def makeAllConstraints() = {
		Console.println("Initializing constraints")
		Console.println("\tForce no degenerate boundary")
		val bPCons = boundaryPositive()
		
		Console.println("\tForce to interior")
		val bIBCons = boxesInsideBoundary()
		
		Console.println("\tForce no intersections")
		val bNICons = noIntersections()

		val allCons = bPCons ++ bIBCons ++ bNICons				
		allCons
	}
	
	def sideLengthObjectives(boxVars:Map[String,CPIntVar], boxTargets:Map[String,Int]):List[CPIntVar] = {
		val boxSides:List[CPIntVar] = boxVars.view.map{
			case(name, varV) =>
				(varV - boxTargets(name)).abs
		}.toList
		
		boxSides
	}
	
	def centerObjectives(boxVars:Map[String, CPIntVar], boxTargets:Map[String, Int]):List[CPIntVar] = {
		val boxCents:List[CPIntVar] = dims.map{
			dim => 
				val dMinV = boxVars(dim + "Min")
				val dMaxV = dMinV + boxVars(dim + "Sz")
				val dMinT = boxTargets(dim + "Min")
				val dMaxT = dMinT + boxTargets(dim + "Sz")
				val centV:CPIntVar = (dMinV + dMaxV)
				val centT = (dMinT + dMaxT)
				(centV - centT).abs
		}
		boxCents
	}
	
	def buildObjective():CPIntVar = {
		val boxSideObjectives = objVTStream.map{
			case(boxVars, boxTargets) => sideLengthObjectives(boxVars.filterKeys(_.endsWith("Sz")), boxTargets.filterKeys(_.endsWith("Sz")))
		}.flatten.toList
		
		val boxCentObjectives = objVTStream.map{
			case(boxVars, boxTargets) => centerObjectives(boxVars, boxTargets)
		}.flatten.toList
		val allObjectives:List[CPIntVar] = sideLengthObjectives(boundaryVars, boundaryTargets) ++ boxSideObjectives ++ boxCentObjectives
		allObjectives.reduceLeft((a,b) => a + b)
	}
	
	def initializeVars() = {
		val varStream = 
		(Stream((boundaryVars("xMax") == snapX(boundaryTargets("xMax"))),
				(boundaryVars("yMax") == snapY(boundaryTargets("yMax")))) ++
		objVTStream.map{
			case(vars, targs) =>
				vars.view.map{
					case(varN, varV) =>
						(varV == snapByName(targs(varN),varN)) // might get some duplicates here, but should be okay, since it'll be the same assignment
				}
		}.flatten)
		varStream.toList
	}
	
	def trySolve(runs:Int = 300, fails:Int = 60, prob:Int = 50) = {
		minimize(objectiveFunction)
		Console.println("Initiating search")
		val probVars = boundaryVars.values.toSeq ++ objStream.map(boxes => boxes.values).flatten.toSeq
		val numVars = probVars.size
		val bestSol = new Array[Int](numVars)
		search {
			binaryFirstFail(probVars)
		} onSolution {
			Console.println("Solution found")
			/*probVars.zipWithIndex.foreach{
				case(v,i) => Console.println("\t => v(" + i + ") = " + v +","+v.value)
			}*/
			(0 until numVars).foreach(i => bestSol(i) = probVars(i).value)
			solved = true
		}
		var stats = /*start(nSols = 1)*/startSubjectTo(nSols = 1){
			add(allCons)
			add(initializeVars())
		}
		Console.println(stats)
		val rand = new Random()
		for(r <- 1 to runs){
			Console.println("Performing next run: " + r)
			stats = startSubjectTo(failureLimit = fails){
				add(allCons)
				add((0 until numVars).filter(i => rand.nextInt(100) < prob).map(i => probVars(i) == bestSol(i)))
			}
			Console.println(stats)
		}
		Console.println("Search complete. Found: ")
		bestSol.zipWithIndex.foreach{
			case(b, i) => probVars(i).assign(b)
		}
		probVars.zipWithIndex.foreach{
			case(v,i) => Console.println("\t => v(" + i + ") = " + v.value)
		}
		stats
	}
	
	def getSolvedObjs() = {
		objVars.map{
			case(labelName,clusters) => 
				val dLVs = objDVars(labelName)
				(labelName -> clusters.map{ 
					case(clusterNumber, boxes) => 
						val dCVs = dLVs(clusterNumber)
						Console.println(labelName + " " + clusterNumber + " (out)")
						(clusterNumber -> boxes.map{
							box => 
								val outRect = new Rect(
										new Point(box("xMin").value, box("yMin").value),
										new Size(dCVs("xSz").value, dCVs("ySz").value)) 
								Console.println("\t" + outRect)
								outRect
						})})}
	}
	
	def getSolvedBoundary() = {
		val outSize = new Size(boundaryVars("xMax").value, boundaryVars("yMax").value)
		Console.println("Found boundary: " + outSize)
		outSize
	}
	
	val allCons = makeAllConstraints
	val objectiveFunction = buildObjective
	
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

class Gridify(val pixel2GridRatio:(Int,Int), val gridStep:(Int, Int)) {
	
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