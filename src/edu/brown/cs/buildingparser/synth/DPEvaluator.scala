package edu.brown.cs.buildingparser.synth

import edu.brown.cs.buildingparser.library.Prim2DRect
import edu.brown.cs.buildingparser.library.Prim2DRectsLib
import org.opencv.core.Mat
import edu.brown.cs.buildingparser.library.Transform
import org.opencv.core.Scalar
import edu.brown.cs.buildingparser.library.Translate
import org.opencv.core.Point


class DPEvaluator(gridStep:(Int, Int), k:Double = 1, l:Double = 0.25) {
	val params = Map("k" -> k, "l" -> l)
	
	val noSolutionC:(Double,Option[(List[Scalar],Prim2DRect)]) = (Double.NegativeInfinity, None)
	val noSolutionR:(Double,Int) = (Double.NegativeInfinity, 0)
	val notInitialized:(Double,Option[(List[Scalar],Prim2DRect)]) = (Double.NaN, None)
	
	class DPHelper(img:Mat, lib:List[Prim2DRect], colorTable:Set[Scalar], colsTable:Array[Array[Array[(Double,Option[(List[Scalar],Prim2DRect)])]]], rowsTable:Array[(Double,Int)]) {
		def optimizeColumns(r:Int, h:Int, w:Int):(Double,Option[(List[Scalar],Prim2DRect)]) = {
			if(h == 0){
				throw new IllegalStateException("Tried to consider 0-height primitives")
			} else if(colsTable(r)(h)(w)._1.isNaN){
				colsTable(r)(h)(w) = if(w == 0){
					(0, None)
				} else {
					colorTable.view.map{
						color =>
							colorTable.view.map{
								complColor =>	
									lib.view.filter(brick => brick.data.width <= w && brick.data.height == h).map{
										brick => 
											val colorVars = List(color, complColor)
											val nextW = w - brick.data.width
											val brickReward = brick.reward(img, colorVars, Some(new Translate(new Point(nextW, r))), params)
											val restReward = optimizeColumns(r, h, nextW)
											(brickReward + restReward._1, Some(colorVars, brick))
									}
							}.flatten
					}.flatten.foldLeft(noSolutionC){
						case(b, a) =>
							if(a._1 > b._1){
								a
							} else {
								b
							}
					}
				}
			}
			colsTable(r)(h)(w)
		}
	
		def optimizeRows(h:Int):(Double,Int) = {
			if(rowsTable(h)._1.isNaN){
				rowsTable(h) = if(h == 0){
					(0, 0)
				} else {
					lib.view.map(brick => brick.data.height).filter(_ <= h).toSet[Int].map{
						brickH =>
							val rowReward = optimizeColumns(h - brickH, brickH, img.width)
							val restReward = optimizeRows(h - brickH)
							// We want to iterate over this whole row!
							(rowReward._1 + restReward._1, brickH)
					}.foldLeft(noSolutionR){
						case(b, a) =>
							if(a._1 > b._1){
								a
							} else {
								b
							}
					}
				}
			}
			rowsTable(h)
		}	
		
		def unpackRow(r:Int, h:Int, c:Int):Stream[(Transform,List[Scalar],Prim2DRect)] = {
			val (bestReward, right) = optimizeColumns(r, h, c)
			right match{
				case Some(primSpec) => 
				case None => throw new IllegalStateException("Was unable to generate a valid tiling with these inputs")
			}
		}
		
		def unpackTable(r:Int):Stream[(Transform,List[Scalar],Prim2DRect)] = {
			val (bestReward, bottomH) = optimizeRows(r)
			if(bottomH == 0){
				
			}
		}
	}
	
	def evaluate(img:Mat, lib:List[Prim2DRect], colorTable:Set[Scalar]):Stream[(Transform,List[Scalar],Prim2DRect)] = {
		(lib.view.map(prim => prim.data) :+ img).foreach{
			mat => if(mat.width % gridStep._1 != 0 || mat.height % gridStep._2 != 0){
				throw new IllegalStateException("Can't run DP unless all image sizes are multiples of " + gridStep)
			}
		}
		val steppedRowCount = img.height / gridStep._2
		val steppedColCount = img.width / gridStep._1
		val colsTable = Array.fill[(Double,Option[(List[Scalar],Prim2DRect)])](steppedRowCount + 1, steppedRowCount + 1, steppedColCount + 1)(notInitialized)
		val rowsTable = Array.fill[(Double,Option[(List[Scalar],Prim2DRect)])](steppedColCount + 1, steppedRowCount + 1)(notInitialized)
	
		val helper = new DPHelper(img:Mat, lib:List[Prim2DRect], colorTable:Set[Scalar], colsTable, rowsTable)
		
		// This invocation is definitely wrong, because we're mixing coordinate spaces
		//val (bestScore, bottomRight) = helper.optimizeRows(img.width, img.height)
		helper.unpackTable(img.height)
		
	}
	
}