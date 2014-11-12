package edu.brown.cs.buildingparser

import org.opencv.core.Core
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.awt.Image
import javax.swing.ImageIcon
import javax.swing.JFrame
import java.awt.FlowLayout
import javax.swing.JLabel
import org.opencv.core.MatOfFloat
import org.opencv.objdetect.HOGDescriptor
import org.opencv.imgproc.Imgproc
import org.opencv.core.Size
import org.opencv.core.Point
import org.opencv.core.CvType
import org.opencv.core.Scalar
import java.io.File
import org.opencv.core.Rect
import jsat.clustering.MeanShift
import jsat.SimpleDataSet
import jsat.classifiers.DataPoint
import collection.JavaConverters._
import jsat.linear.Vec
import jsat.linear.DenseVector
import jsat.classifiers.CategoricalData
import org.opencv.core.MatOfPoint
import jsat.DataSet

object Util {
	
	def calcGrabBox(box:Rect, bounds:Size):Rect = {
		val grabW = bounds.width.toInt
		val grabH = bounds.height.toInt
		val xdiff = grabW - box.width
		val ydiff = grabH - box.height
		val x = box.x - xdiff / 2
		val y = box.y - ydiff / 2
		val grabBox = new Rect(x,y,grabW,grabH)
		grabBox
	}
	
	def bestBoundsBucket(box:Rect, bounds:List[Size]):Size = {
			bounds.filter(bound => 
			bound.width >= box.width && 
			bound.height >= box.height).foldLeft(null.asInstanceOf[Size])(
					(best, proposed) => 
					if(best == null || Math.max(best.width - box.width, best.height - box.height) > Math.max(proposed.width - box.width, proposed.height - box.height)){
						proposed	
					} else {
						best
					})
	}
	
	def dumpClusters(clusters:List[List[DataPoint]]) = {
		clusters.foreach{ cluster =>
			Console.println("Cluster:")
			cluster.foreach{ dp =>
				Console.println("\t" + dp.toString().substring(0,50) + ", ")
			}
			Console.println("")
		}
	}
	
	def clusterRects(bbs:List[Rect]):Array[Int] = {
		Console.println("Clustering rects")
		val clusterer = new MeanShift
		val data = new SimpleDataSet(bbs.map{rect => 
			val tl = rect.tl
			val br = rect.br
			val rectV = new DenseVector(List(tl.x, tl.y, br.x, br.y).map(_.asInstanceOf[java.lang.Double]).asJava)
			val dp = new DataPoint(rectV, new Array[Int](0), new Array[CategoricalData](0))
			dp}.asJava)
		val clusterAssignments = clusterer.cluster(data, null)
		/*
		clusterAssignments.foreach{
			i => Console.print(i + " ")
		}
		Console.println("")
		*/
		clusterAssignments
	}
	
	def calcBandwidth(data:DataSet, scale:Double):Double = {
		val vecs = data.getDataVectors().asScala
		var distCounter = 0.
		for(i <- 0 until vecs.size; j <- 0 until i){
			val distUpd = vecs(i).subtract(vecs(j)).pNorm(2)
			distCounter += distUpd
			//Console.println(distUpd)
		}
		distCounter /= (vecs.length * (vecs.length - 1) / 2)
		(distCounter * scale)
	}
	
	def filterContentsByExts(handle:File, exts:Set[String]):Set[File] = {
			(Set[File]() ++ handle.listFiles).filter(img => exts.exists( ext => img.getName.endsWith(ext) ) )
	}
	
	def filterContentsByIsDir(handle:File, wantDirs:Boolean):Set[File] = {
		(Set[File]() ++ handle.listFiles).filter(handle => handle.isDirectory == wantDirs )
	}
	
	def svmFeatureFromHog(hog:HOGDescriptor, maxHog:HOGDescriptor, hogF:Mat):Mat = {
		val dim = (maxHog.getDescriptorSize() + 2).asInstanceOf[Int]
		val outMat = Mat.zeros(1, dim, CvType.CV_32F)
		val sizeBuf = new Array[Float](1)
		sizeBuf(0) = hog.get_winSize().width.asInstanceOf[Float]
		outMat.put(0,0,sizeBuf)
		sizeBuf(0) = hog.get_winSize().height.asInstanceOf[Float]
		outMat.put(0,1,sizeBuf)
		hogF.t().copyTo(outMat.submat(0,1,2,2+hogF.rows))
		
		/*
		Console.println("svm from hog")
		Console.println(outMat.get(0,0)(0) + " x " + outMat.get(0,1)(0))
		Console.println(hog.get_winSize())
		Console.println(outMat.dump())
		Console.println(hogF.t().dump())
		*/
		outMat
	}
	
	def makeBoundaryFilled(inMat:Mat, border:Int):Mat = {
		val outMat = Mat.zeros(inMat.rows + 2 * border, inMat.cols + 2 * border, inMat.`type`)
		// Copy primary region
		inMat.copyTo(outMat.submat(border,border+inMat.rows,border,border+inMat.cols))
		outMat
	}
	
	def makeBoundaryMirrored(inMat:Mat, border:Int):Mat = {
		val outMat = new Mat(inMat.rows + 2 * border, inMat.cols + 2 * border, inMat.`type`)
		val colBuf = new Mat(inMat.rows,border,inMat.`type`)
		val rowBuf = new Mat(border,inMat.cols,inMat.`type`)
		val cornerBuf = new Mat(border,border,inMat.`type`)
		// Copy primary region
		inMat.copyTo(outMat.submat(border,border+inMat.rows,border,border+inMat.cols))
		
		// Copy edges
		Core.flip(inMat.submat(0,inMat.rows,0,border),colBuf,1)
		colBuf.copyTo(outMat.submat(border,outMat.rows-border,0,border))
		
		Core.flip(inMat.submat(0,inMat.rows,inMat.cols-border,inMat.cols),colBuf,1)
		colBuf.copyTo(outMat.submat(border,outMat.rows-border,outMat.cols-border,outMat.cols))
		
		Core.flip(inMat.submat(0,border,0,inMat.cols),rowBuf,0)
		rowBuf.copyTo(outMat.submat(0,border,border,outMat.cols-border))
		
		Core.flip(inMat.submat(inMat.rows-border,inMat.rows,0,inMat.cols),rowBuf,0)
		rowBuf.copyTo(outMat.submat(outMat.rows-border,outMat.rows,border,outMat.cols-border))
		
		// Copy corners
		Core.flip(inMat.submat(0,border,0,border),cornerBuf,-1)
		cornerBuf.copyTo(outMat.submat(0,border,0,border))
		
		Core.flip(inMat.submat(0,border,inMat.cols-border,inMat.cols),cornerBuf,-1)
		cornerBuf.copyTo(outMat.submat(0,border,outMat.cols-border,outMat.cols))
		
		Core.flip(inMat.submat(inMat.rows-border,inMat.rows,inMat.cols-border,inMat.cols),cornerBuf,-1)
		cornerBuf.copyTo(outMat.submat(outMat.rows-border,outMat.rows,outMat.cols-border,outMat.cols))
		
		Core.flip(inMat.submat(inMat.rows-border,inMat.rows,0,border),cornerBuf,-1)
		cornerBuf.copyTo(outMat.submat(outMat.rows-border,outMat.rows,0,border))
		
		outMat
	}
	
	def matToImage(inMat:Mat):BufferedImage = {
		val kind = inMat.channels match {
			case 1 => BufferedImage.TYPE_BYTE_GRAY
			case 3 => BufferedImage.TYPE_3BYTE_BGR
			case 4 => Console.println("Warning 4 bytes")
			BufferedImage.TYPE_4BYTE_ABGR
			case _ => throw new java.awt.AWTException("Don't know what to do with this channel layout")
		}
		
		val bufferSize = inMat.channels * inMat.cols * inMat.rows 
		val b = new Array[Byte](bufferSize)
		inMat.get(0,0,b)
		val image = new BufferedImage(inMat.cols,inMat.rows,kind)
		System.arraycopy(b, 0, image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData, 0, b.length)
		image
	}
	
	def visHogF(img:Mat, hog:HOGDescriptor, descriptor:MatOfFloat, scale:Int = 8, viz_factor:Int = 4):Mat = {
		val descriptorValues = descriptor.toArray
		val outImg:Mat = new Mat
		Imgproc.resize(img, outImg, new Size(img.cols*scale,img.rows*scale))
		// Mat.zeros(img.cols*scale,img.rows*scale,img.`type`)
		val cmpImg:Mat = outImg.clone
		
		
		val gradientBinSize = hog.get_nbins()
		val winSize = hog.get_winSize()
		val cellSize = hog.get_cellSize()
		val blockSize = hog.get_blockSize()
		
		val cells_per_block_x = (blockSize.width / cellSize.width).toInt
		val cells_per_block_y = (blockSize.height / cellSize.height).toInt 
		val blockStride = hog.get_blockStride()
		val radRangeForOneBin = 3.14/gradientBinSize
		
		val cells_in_x_dir = (winSize.width / cellSize.width).toInt
		val cells_in_y_dir = (winSize.height / cellSize.height).toInt
		val totalnrofcells = cells_in_x_dir * cells_in_y_dir
		
		val gradientStrengths = new Array[Array[Array[Float]]](cells_in_y_dir)
		val cellUpdateCounter = new Array[Array[Int]](cells_in_y_dir)
		for(y <- 0 to (cells_in_y_dir - 1)) {
			gradientStrengths(y) = new Array[Array[Float]](cells_in_x_dir)
			cellUpdateCounter(y) = new Array[Int](cells_in_x_dir)
			for(x <- 0 to (cells_in_x_dir - 1)){
				gradientStrengths(y)(x) = new Array[Float](gradientBinSize)
				cellUpdateCounter(y)(x) = 0
				for(bin <- 0 to (gradientBinSize - 1)){
					gradientStrengths(y)(x)(bin) = 0
				}
			}
		}
		
		
		
		val blocks_in_x_dir = cells_in_x_dir - 1 // This number isn't general, need to think about it more!
		val blocks_in_y_dir = cells_in_y_dir - 1 // This number isn't general, need to think about it more!
		val totalnrofBlocks = blocks_in_x_dir * blocks_in_y_dir
		
		for(blockX <- 0 to (blocks_in_x_dir - 1)){
			for(blockY <- 0 to (blocks_in_y_dir - 1)){
				for(cellX <- 0 to (cells_per_block_x - 1)){
					for(cellY <- 0 to (cells_per_block_y -1 )){
						for(bin <- 0 to (gradientBinSize - 1)){
							/* 
							 val descriptorDataIdx	= blockX 	* blocks_in_y_dir	* cells_per_block_x	* cells_per_block_y	* gradientBinSize 
																		+ blockY	* cells_per_block_x	* cells_per_block_y	* gradientBinSize
																						+ cellX			* cells_per_block_y	* gradientBinSize
																											+ cellY			* gradientBinSize
																															+ bin
																															*/
							val descriptorDataIdx = bin + gradientBinSize * (cellY + cells_per_block_y * (cellX + cells_per_block_x * (blockY + blocks_in_y_dir * blockX)))
							gradientStrengths(blockY + cellY)(blockX + cellX)(bin) += descriptorValues(descriptorDataIdx)
							
						}
						cellUpdateCounter(blockY + cellY)(blockX + cellX) += 1
					}
				}
			}
		}
		
		for(cellY <- 0 to (cells_in_y_dir - 1)){
			for(cellX <- 0 to (cells_in_x_dir - 1)){
				val NrUpdatesForThisCell = cellUpdateCounter(cellY)(cellX)
				for(bin <- 0 to (gradientBinSize - 1)) {
					gradientStrengths(cellY)(cellX)(bin) /= NrUpdatesForThisCell
				}
			}
		}
	
		
		for(cellY <- 0 to (cells_in_y_dir - 1)){
			for(cellX <- 0 to (cells_in_x_dir - 1)){
				val drawX = cellX * cellSize.width
				val drawY = cellY * cellSize.height
				
				val mx = drawX + cellSize.width / 2
				val my = drawY + cellSize.height / 2
				
				Core.rectangle(outImg, 
						new Point(drawX * scale,drawY * scale), 
						new Point((drawX + cellSize.width) * scale, (drawY + cellSize.height) * scale), 
						new Scalar(100),//,100,100),
						2)
						
				for(bin <- 0 to (gradientBinSize - 1)){
					val currentGradStrength = gradientStrengths(cellY)(cellX)(bin)
					if(currentGradStrength != 0){
						val currad = (bin + 0.5) * radRangeForOneBin
						val dirVecX = Math.cos(currad)
						val dirVecY = Math.sin(currad)
						val maxVecLen = cellSize.width / 2
						val x1 = mx - dirVecX * currentGradStrength * maxVecLen * viz_factor
						val y1 = my - dirVecY * currentGradStrength * maxVecLen * viz_factor
						val x2 = mx + dirVecX * currentGradStrength * maxVecLen * viz_factor
						val y2 = my + dirVecY * currentGradStrength * maxVecLen * viz_factor
						Core.line(outImg,
								new Point(x1*scale,y1*scale),
								new Point(x2*scale,y2*scale),
								new Scalar(255),//0,0,255),
                     2)
					}
				}
			}
		}/*
		for(r <- 0 to (outImg.rows -1 )){
			for(c <- 0 to (outImg.cols -1 )){
				val data = new Array[Array[Byte]](3)
				data(0) = new Array[Byte](1)
				outImg.get(r,c,data(0))
				data(1) = new Array[Byte](1)
				cmpImg.get(r,c,data(0))
				data(2) = new Array[Byte](1)
				data(2)(0) = (data(1)(0) - data(0)(0)).asInstanceOf[Byte]
				outImg.put(r,c,data(2))
			}
		}*/
		outImg
	}
	
	def makeImageFrame(img:Image, title:String = ""):Unit = {
		val icon = new ImageIcon(img)
		val frame=new JFrame(title)
		frame.setLayout(new FlowLayout);       
		frame.setSize(icon.getIconWidth + 50, icon.getIconHeight + 50);
		frame.add(new JLabel(icon))
		frame.setVisible(true)
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
	}

}