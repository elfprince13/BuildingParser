package edu.brown.cs.buildingparser.synth

import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.core.Point
import edu.brown.cs.buildingparser.Util

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
	
	def cutWithRect(cutter:Rect, cuttees:List[Rect]):(List[Rect], List[Rect]) = {
		val (inCuts, outCuts) = cuttees.foldLeft((List[Rect](), List[Rect]())){
			case((inCuts, outCuts), cuttee) =>
				Util.checkContains(cutter, cuttee) match {
					case Some(interior:Rect) =>
						val topDone = if(cuttee.tl.y < interior.tl.y){
							(new Rect(cuttee.tl, new Point(cuttee.br.x, interior.tl.y - 1))) :: outCuts
						} else { 
							outCuts 
						} 
						val leftDone = if(cuttee.tl.x < interior.tl.x){
							(new Rect(new Point(cuttee.tl.x, interior.tl.y), new Point(interior.tl.x - 1, interior.br.y))) :: topDone
						} else { 
							topDone
						} 
						val rightDone = if(interior.br.x < cuttee.br.x){
							(new Rect(new Point(interior.br.x + 1, interior.tl.y), new Point(cuttee.br.x, interior.br.y))) :: leftDone
						} else { 
							leftDone 
						} 
						val allDone = if(interior.br.y < cuttee.br.y){
							(new Rect(new Point(cuttee.tl.x, interior.br.y + 1), cuttee.br)) :: rightDone
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
	def getNonObjRegions(bounds:Rect, sortedObjs:List[Rect]):List[Rect] = {
		if(sortedObjs.size == 0){
			List(bounds)
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
						if(divHere.tl.y <= thisDiv.br.y){
							if(divHere.br.x < thisDiv.tl.x){
								(divHere :: left, bottom, right)
							} else if(thisDiv.br.x < divHere.tl.x){
								(left, bottom, divHere :: right)
							} else {
								throw new IllegalStateException("Unwanted touch: " + thisDiv + " is overlapped by " + divHere)
							}
						} else {
							(left, divHere :: bottom, right)
						}
				}
				
				val lRect = new Rect(bounds.tl, new Point(thisDiv.tl.x - 1, thisDiv.br.y))
				val bRect = new Rect(new Point(bounds.tl.x, thisDiv.br.y + 1), bounds.br)
				val rRect = new Rect(new Point(thisDiv.br.x + 1, bounds.tl.y), new Point(bounds.br.x, thisDiv.br.y))
				val (lAppend, lRemains) = {
					val (in, out) = cutWithRect(lRect, left)
					(getNonObjRegions(lRect, in), out)
				}
				
				val (rAppend, rRemains) = {
					val (in, out) = cutWithRect(rRect, right)
					(getNonObjRegions(rRect, in), out)
				}
				
				sortObjs(lAppend ::: rAppend) ::: getNonObjRegions(bRect, sortObjs(lRemains ::: rRemains ::: bottom))
			} else if(thisDiv.tl.y > bounds.tl.y){
				 (new Rect(bounds.tl, new Point(bounds.br.x,thisDiv.tl.y - 1))) :: getNonObjRegions(new Rect(new Point(bounds.tl.x,thisDiv.tl.y), bounds.br),sortedObjs)
			} else {
				throw new IllegalStateException("Containment breach: " + thisDiv + " leaks from " + bounds)
			}
		}
	}
}