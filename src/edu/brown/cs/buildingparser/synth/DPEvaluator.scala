package edu.brown.cs.buildingparser.synth

import edu.brown.cs.buildingparser.library.Prim2DRect
import edu.brown.cs.buildingparser.library.Prim2DRectsLib
import edu.brown.cs.buildingparser.library.BrickSynth
import org.opencv.core.Mat
import edu.brown.cs.buildingparser.library.Transform
import org.opencv.core.Scalar
import edu.brown.cs.buildingparser.library.Translate
import org.opencv.core.Point

import org.opencv.highgui.Highgui
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc
import edu.brown.cs.buildingparser.Util
import edu.brown.cs.buildingparser.training.SamplerMain


class DPEvaluator(gridStep:(Int, Int), k:Double = 1, l:Double = 0.25) {
	val params = Map("k" -> k, "l" -> l)
	
	val noSolutionC:(Double,Option[(List[Scalar],Prim2DRect)]) = (Double.NegativeInfinity, None)
	val noSolutionR:(Double,Int) = (Double.NegativeInfinity, 0)
	val notInitializedC:(Double,Option[(List[Scalar],Prim2DRect)]) = (Double.NaN, None)
	val notInitializedR:(Double,Int) = (Double.NaN, 0)
	
	class DPHelper(img:Mat, lib:List[Prim2DRect], colorTable:Set[(Scalar,Scalar)], colsTable:Array[Array[Array[(Double,Option[(List[Scalar],Prim2DRect)])]]], rowsTable:Array[(Double,Int)]) {
		val heightSetsLE = new Array[Set[Int]](rowsTable.length) 
		val brickSetsE = new Array[Array[List[Prim2DRect]]](rowsTable.length)
		Console.println("Initializing height sets")
		(0 until heightSetsLE.length).foreach{
			h => 
				Console.println(f"height $h")
				heightSetsLE(h) = lib.view.map(brick => brick.data.height / gridStep._2).filter(_ <= h).toSet[Int]
				brickSetsE(h) = new Array[List[Prim2DRect]](colsTable(h)(h).length)
				(0 until brickSetsE(h).length).foreach{
					w =>
						Console.println(f"  width $w")
						brickSetsE(h)(w) = lib.view.filter(brick => brick.data.width / gridStep._1 <= w && brick.data.height / gridStep._2 == h).toList
				}
		}
		Console.println("Done Initializing")
		
		def optimizeColumns(r:Int, h:Int, w:Int):(Double,Option[(List[Scalar],Prim2DRect)]) = {
			if(h == 0){
				throw new IllegalStateException("Tried to consider 0-height primitives")
			} else if(colsTable(r)(h)(w)._1.isNaN){
				//Console.println(f"  table cell ($r)($h)($w) is not cached")
				colsTable(r)(h)(w) = if(w == 0){
					(0, None)
				} else {
					colorTable.view.map{
						case(color,complColor) =>
							brickSetsE(h)(w).map{
								brick => 
									val colorVars = List(color, complColor)
									val nextW = w - (brick.data.width / gridStep._1)
									val brickReward = brick.reward(img, colorVars, Some(new Translate(new Point(nextW * gridStep._1, r * gridStep._2))), params)
									val restReward = optimizeColumns(r, h, nextW)
									(brickReward + restReward._1, Some(colorVars, brick))
							}
					}.flatten.foldLeft(noSolutionC){
						case(b, a) =>
							if(a._1 == b._1){
								Console.println(f"weird, there's a tie! ${a._1} == ${b._1}")
							}
							if(a._1 > b._1){
								a
							} else {
								b
							}
					}
				}
				//Console.println(f"  table cell ($r)($h)($w) is now cached")
			}
			colsTable(r)(h)(w)
		}
	
		def optimizeRows(h:Int):(Double,Int) = {
			if(rowsTable(h)._1.isNaN){
				Console.println(f"table row $h is not cached")
				rowsTable(h) = if(h == 0){
					(0, 0)
				} else {
					heightSetsLE(h).map{
						brickH =>
							val rowReward = optimizeColumns(h - brickH, brickH, img.width / gridStep._1)
							val restReward = optimizeRows(h - brickH)
							// We want to iterate over this whole row!
							(rowReward._1 + restReward._1, brickH)
					}.foldLeft(noSolutionR){
						case(b, a) =>
							if(a._1 == b._1){
								Console.println(f"weird, there's a tie! ${a._1} == ${b._1}")
							}
							if(a._1 > b._1){
								a
							} else {
								b
							}
					}
				}
				Console.println(f"table row $h is now cached")
			}
			rowsTable(h)
		}	
		
		def unpackRow(r:Int, h:Int, c:Int):Stream[(Transform,List[Scalar],Prim2DRect)] = {
			Console.println(f"Unpacking row: ($r)($h)($c)")
			val (bestReward, right) = optimizeColumns(r, h, c)
			right match{
				case Some(primSpec) => 
					val nextC = c - (primSpec._2.data.width / gridStep._1)
					(new Translate(new Point(nextC * gridStep._1, r * gridStep._2)), primSpec._1, primSpec._2) #:: unpackRow(r, h, nextC)
				case None => 
					if(c == 0){
						Stream[(Transform,List[Scalar],Prim2DRect)]()
					} else {
						throw new IllegalStateException("Was unable to generate a valid tiling with these inputs")
					}
			}
		}
		
		def unpackTable(bottomR:Int):Stream[(Transform,List[Scalar],Prim2DRect)] = {
			Console.println(f"Unpacking table: ($bottomR)")
			val (bestReward, h) = optimizeRows(bottomR)
			val r = bottomR - h
			if(r == 0 && h == 0){
				Stream[(Transform,List[Scalar],Prim2DRect)]()
			} else if(h != 0){
				unpackRow(r, h, img.width / gridStep._1) #::: unpackTable(r)
			} else {
				throw new IllegalStateException("Was unable to generation a valid tiling with these inputs (got h == 0 && r != 0")
			}
		}
	}
	
	def evaluate(img:Mat, lib:List[Prim2DRect], colorTable:Set[(Scalar,Scalar)]):Stream[(Transform,List[Scalar],Prim2DRect)] = {
		(lib.view.map(prim => prim.data) :+ img).foreach{
			mat => if(mat.width % gridStep._1 != 0 || mat.height % gridStep._2 != 0){
				throw new IllegalStateException(f"Can't run DP unless all image sizes are multiples of $gridStep but got (${mat.width}, ${mat.height}) instead")
			}
		}
		val steppedRowCount = img.height / gridStep._2
		val steppedColCount = img.width / gridStep._1
		val colsTable = Array.fill[(Double,Option[(List[Scalar],Prim2DRect)])](steppedRowCount + 1, steppedRowCount + 1, steppedColCount + 1)(notInitializedC)
		val rowsTable = Array.fill[(Double,Int)](steppedRowCount + 1)(notInitializedR)
		Console.println(f"rows table has ${rowsTable.length} entries")
		Console.println(f"cols table has ${(steppedRowCount + 1) * (steppedRowCount + 1) * (steppedColCount + 1)} entries")
		Console.println(f"${colorTable.size * lib.length} brick / color combos")
	
		val helper = new DPHelper(img:Mat, lib:List[Prim2DRect], colorTable:Set[(Scalar,Scalar)], colsTable, rowsTable)
		helper.unpackTable(img.height / gridStep._2)
	}	
}

object DPTest {
	def main(args:Array[String]):Unit = {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
		SamplerMain.isMain = false
		val grazSampler = SamplerMain.grazSampler
		
		val srcHandle = grazSampler.imgHandleFromName("facade_1_0056092_0056345.png")//("facade_0_0099003_0099285.png")//
		//val labelHandle = grazSampler.pairedImgs(srcHandle)
		//val (srcBase, examples) = grazSampler.extractOneExampleSet(srcHandle, labelHandle)
		val srcImg = Highgui.imread(srcHandle.getAbsolutePath, Highgui.CV_LOAD_IMAGE_COLOR)
		val gridRows = LDrawGridify.nearestMeshPoint(srcImg.rows, LDrawGridify.gridStep._2)
		val gridCols = LDrawGridify.nearestMeshPoint(srcImg.cols, LDrawGridify.gridStep._1)
		
		val gridImg = new Mat(gridRows, gridCols, srcImg.`type`)
		Imgproc.resize(srcImg, gridImg, gridImg.size)
		Console.println(f"Resizing ${srcImg.cols} x ${srcImg.rows} to $gridCols x $gridRows for ${LDrawGridify.gridStep}")
		
		val brickPlacer = new DPEvaluator(LDrawGridify.gridStep)
		val instr = brickPlacer.evaluate(gridImg, BrickSynth.getStdBricks(), BrickSynth.COLOR_TABLE.toSet[(Scalar,Scalar)])
		
		val dstImg = Mat.zeros(gridImg.size, gridImg.`type`)
		instr.foreach{
			case(proj,colors,brick) =>
				brick.project(dstImg, colors, Some(proj))
		}
		
		Util.makeImageFrame(Util.matToImage(srcImg), "src")
		Util.makeImageFrame(Util.matToImage(gridImg), "stretched to grid")
		Util.makeImageFrame(Util.matToImage(dstImg), "bricked over")
		
		
		
	}
}