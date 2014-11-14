package edu.brown.cs.buildingparser.synth

import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.core.Point
import edu.brown.cs.buildingparser.Util
import org.opencv.core.Mat
import scala.util.Random
import org.opencv.core.Scalar
import org.opencv.core.Core
import edu.brown.cs.buildingparser.Main

class DPSubdivider(gridStep:(Int, Int)) {
	val xStep = gridStep._1
	val yStep = gridStep._2
	
	def sortObjs(objects:List[Rect]):List[Rect] = objects.sortWith(orderRects)
	def orderRects(r1:Rect, r2:Rect):Boolean = orderPoints(r1.tl, r2.tl)
	def orderPoints(p1:Point, p2:Point):Boolean = {
		if(p1.y < p2.y){
			true
		} else if(p1.y > p2.y){
			false
		} else {
			p1.x < p2.x
		}		
	}
	val rg = new Random
	
	def safeRect(p1:Point, p2:Point) = {
		val rect = new Rect(p1, p2)
		if(rect.width < xStep || rect.height < yStep){
			throw new IllegalStateException("shouldn't create rects small than the grid: " + rect)
		} else{
			rect
		}
	}
	
	def cutWithRect(cutter:Rect, cuttees:List[Rect]):(List[Rect], List[Rect]) = {
		val (inCuts, outCuts) = cuttees.foldLeft((List[Rect](), List[Rect]())){
			case((inCuts, outCuts), cuttee) =>
				Util.checkContains(cutter, cuttee) match {
					case Some(interior:Rect) =>
						val topDone = if(cuttee.tl.y < interior.tl.y){
							(safeRect(cuttee.tl, new Point(cuttee.br.x, interior.tl.y))) :: outCuts
						} else { 
							outCuts 
						} 
						val leftDone = if(cuttee.tl.x < interior.tl.x){
							(safeRect(new Point(cuttee.tl.x, interior.tl.y), new Point(interior.tl.x, interior.br.y))) :: topDone
						} else { 
							topDone
						} 
						val rightDone = if(interior.br.x < cuttee.br.x){
							(safeRect(new Point(interior.br.x, interior.tl.y), new Point(cuttee.br.x, interior.br.y))) :: leftDone
						} else { 
							leftDone 
						} 
						val allDone = if(interior.br.y < cuttee.br.y){
							(safeRect(new Point(cuttee.tl.x, interior.br.y), cuttee.br)) :: rightDone
						} else { 
							rightDone
						} 
						(interior :: inCuts, allDone)
					case None => (inCuts, cuttee :: outCuts)
				}
		}
		// Too tired to decide if this is necessary:
		(sortObjs(inCuts), sortObjs(outCuts))
		
	}
	
	/*
	 *  Precondition: sortedObjs are:
	 *  (a) contained entirely in bounds
	 *  (b) sorted by orderRects
	 *  (c) have integral components
	 *  (d) don't overlap
	 *  Postcondition: output is:
	 *  (a) the complement of sortedObjs
	 *  (b) sorted by orderRects / ready for DPing
	 */
	def getNonObjRegions(bounds:Rect, sortedObjs:List[Rect], debugImg:Option[Mat] = None):List[Rect] = {
		debugImg match {
			case Some(img) => Util.makeImageFrame(Util.matToImage(img), bounds.toString)
			case None =>
		}
		if(sortedObjs.size == 0){
			debugImg match {
				case Some(img) => Core.rectangle(img, bounds.tl, Main.ofsPoint(bounds.br, new Point(-1, -1)), new Scalar(rg.nextInt(256), rg.nextInt(256), rg.nextInt(256)), -1)
				case None =>
			}
			if(bounds.width == 0 || bounds.height == 0){
				List()
			} else{
				List(bounds)
			}
		} else {
			val thisDiv = sortedObjs.head
			val others = sortedObjs.tail
			if(thisDiv.tl.y == bounds.tl.y){
				/*
				 * Left/right/bottom subdivision.
				 * This is tricky business
				 */
				
				val (left, bottom, right) = others.foldLeft((List[Rect](),List[Rect](),List[Rect]())){
					case((left, bottom, right), divHere) =>
						if(divHere.tl.y < thisDiv.br.y){
							if(divHere.br.x <= thisDiv.tl.x){
								(divHere :: left, bottom, right)
							} else if(thisDiv.br.x <= divHere.tl.x){
								(left, bottom, divHere :: right)
							} else {
								throw new IllegalStateException("Unwanted touch: " + thisDiv + " is overlapped by " + divHere)
							}
						} else {
							(left, divHere :: bottom, right)
						}
				}
				
				val lRect = new Rect(bounds.tl, new Point(thisDiv.tl.x, thisDiv.br.y))
				val bRect = new Rect(new Point(bounds.tl.x, thisDiv.br.y), bounds.br)
				val rRect = new Rect(new Point(thisDiv.br.x, bounds.tl.y), new Point(bounds.br.x, thisDiv.br.y))
				
				// Need to use the cuts to bound regions
				val (lAppend, lRemains) = if(lRect.width > 0 && lRect.height > 0){
					val (in, out) = cutWithRect(lRect, left)
					(getNonObjRegions(lRect, in, debugImg), out)
				} else {
					if(left.size != 0){
						throw new IllegalStateException("found objects in empty region, " + lRect + ": " + left)
					} else {
						(List(), List())
					}
				}
				
				val (rAppend, rRemains) = if(rRect.width > 0 && rRect.height > 0){
					val (in, out) = cutWithRect(rRect, right)
					(getNonObjRegions(rRect, in, debugImg), out)
				} else {
					if(right.size != 0){
						throw new IllegalStateException("found objects in empty region, " + rRect + ": " + right)
					} else {
						(List(), List())
					}
				}
				
				val outBounds = sortObjs((lAppend ::: rAppend))//.filter(rect => rect.width > 0 && rect.height > 0))
				
				debugImg match {
					case Some(img) => outBounds.map{ rect => Core.rectangle(img, rect.tl, Main.ofsPoint(rect.br, new Point(-1, -1)), new Scalar(rg.nextInt(256), rg.nextInt(256), rg.nextInt(256)), -1) }
					case None =>
				}
				
				if(bRect.width == 0 || bRect.height == 0){
					outBounds
				} else {
					outBounds ::: getNonObjRegions(bRect, sortObjs(lRemains ::: rRemains ::: bottom), debugImg)	
				}
			} else if(thisDiv.tl.y > bounds.tl.y){
				// This might be where things go wrong
				
				val tRect = (safeRect(bounds.tl, new Point(bounds.br.x,thisDiv.tl.y)))
				debugImg match {
					case Some(img) => if(tRect.width > 0 && tRect.height > 0) Core.rectangle(img, tRect.tl, Main.ofsPoint(tRect.br, new Point(-1, -1)), new Scalar(rg.nextInt(256), rg.nextInt(256), rg.nextInt(256)), -1)
					case None =>
				}
				val remRects = getNonObjRegions(safeRect(new Point(bounds.tl.x,thisDiv.tl.y), bounds.br),sortedObjs, debugImg)
				if(tRect.width == 0 || tRect.height == 0){
					remRects
				} else {
					tRect :: remRects
				}
			} else {
				throw new IllegalStateException("Containment breach: " + thisDiv + " leaks from " + bounds)
			}
		}
	}
}