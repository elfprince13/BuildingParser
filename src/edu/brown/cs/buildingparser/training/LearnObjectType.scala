package edu.brown.cs.buildingparser.training

import org.opencv.objdetect.HOGDescriptor
import org.opencv.ml.CvSVM
import java.io.File
import edu.brown.cs.buildingparser.Util
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.highgui.Highgui
import org.opencv.core.MatOfFloat
import scala.actors.Actor
import scala.actors.Channel

class LearnObjectType(className:String, objDir:String, objAntiDir:String, counterClasses:List[String],
		exts:Set[String], hogs:List[HOGDescriptor]) {
	val hogBuckets = (hogs.seq.map(hog => hog.get_winSize() -> hog)).toMap
	val biggestBucket = hogBuckets.keys.maxBy(_.area)
	
	var mySVM:CvSVM = null
	var posExamples:List[String] = null
	var negExamples:List[String] = null
	var totalExamples = 0
	
	var trainData:Mat = null
	var responses:Mat = null
	
	def initSVM():Unit = {
		mySVM = new CvSVM()
	}
	
	def initExamplesLists():Unit = {
		val objSrcHandle = new File(objDir + File.separator + className)
		val antiObjSrcHandle = new File(objAntiDir + File.separator + className)
		val counterObjSrcHandles = counterClasses.map(otherClass => new File(objDir + File.separator + otherClass))
		posExamples = Util.filterContentsByExts(objSrcHandle, exts).map(handle => handle.getAbsolutePath).toList
		negExamples = Util.filterContentsByIsDir(antiObjSrcHandle, true).map(handle => 
			Util.filterContentsByExts(handle, exts)).flatten.map(handle => handle.getAbsolutePath).toList
		negExamples ++= counterObjSrcHandles.map(counterHandle =>
			Util.filterContentsByExts(counterHandle, exts)).flatten.map(handle => handle.getAbsolutePath)
		Console.println(posExamples.size + " positive examples, and " + negExamples.size + " negative examples for " + className)
		totalExamples = posExamples.size + negExamples.size
	}
	
	def initTrainingData():Unit = {
		val posCount = posExamples.size
		val negCount = negExamples.size
		val maxHog = hogBuckets(biggestBucket)
		val rowWidth = (2 + maxHog.getDescriptorSize).asInstanceOf[Int]
		trainData = Mat.zeros(totalExamples, rowWidth, CvType.CV_32F)
		responses = Mat.zeros(totalExamples, 1 , CvType.CV_32F)
		Mat.ones(posCount, 1, CvType.CV_32F).copyTo(responses.submat(0, posCount, 0, 1))
		Mat.ones(negCount, 1, CvType.CV_32F).mul(Mat.ones(negCount, 1, CvType.CV_32F), -1).copyTo(responses.submat(posCount, totalExamples, 0, 1))
		
		Console.println("Estimate trainData for " + className + " is " + ((trainData.size.area * 4) / (1024 * 1024)).intValue + "MB")
		Console.println("Copying posExamples for " + className + " ...\t")
		posExamples.seq.zipWithIndex.foreach{
			case(posExample, row) =>
				val exMat = Highgui.imread(posExample, Highgui.CV_LOAD_IMAGE_GRAYSCALE)
				val hog = hogBuckets(exMat.size)
				val exHogF = new MatOfFloat
				hog.compute(exMat,exHogF)
				Util.svmFeatureFromHog(hog, maxHog, exHogF).copyTo(trainData.submat(row, row + 1, 0, rowWidth))
		}
		Console.println("Copying negExamples for " + className + " ...\t")
		negExamples.seq.zipWithIndex.foreach{
			case(negExample, row) =>
				val exMat = Highgui.imread(negExample, Highgui.CV_LOAD_IMAGE_GRAYSCALE)
				val hog = hogBuckets(exMat.size)
				val exHogF = new MatOfFloat
				hog.compute(exMat,exHogF)
				Util.svmFeatureFromHog(hog, maxHog, exHogF).copyTo(trainData.submat(posCount + row, posCount + row + 1, 0, rowWidth))
		}
	}
	
	def clearExamplesLists():Unit = {
		posExamples = null
		negExamples = null
	}
	
	def doTrain():Unit = {
		Console.println("Beginning training of " + className)
		mySVM.train(trainData, responses)
	}
	
	def saveResults(savedir:String, prefix:String = "", suffix:String = "" , ext:String = "opencv_svm"):Unit = {
		mySVM.save(savedir +  File.separator + prefix + "_" + className + "_" + suffix + "." + ext)
	}
	
}

object LearnerFactory {
	class Learner(className:String, counterClasses:List[String], usedHogs:List[HOGDescriptor]) extends Actor {
		def act():Unit = {
			val learner = new LearnObjectType(className, "object-classes", "anti-object-classes", counterClasses, Set[String](".png",".jpg",".jpeg"), usedHogs)
			learner.initExamplesLists
			learner.initTrainingData
			learner.clearExamplesLists // Save some memory
			learner.initSVM
			learner.doTrain
			learner.saveResults("trained-svms", prefix = ("" + System.currentTimeMillis))
			Console.println(className + "-learner complete")
			exit
		}
	}
	
	def learnDoors(usedHogs:List[HOGDescriptor]) = {
		val doorLearner = new Learner("door", List("window","balcony"), usedHogs)
		doorLearner.act
		doorLearner
	}
	
	def learnWindows(usedHogs:List[HOGDescriptor]) = {
		val windowLearner = new Learner("window",List("door","balcony"),usedHogs)
		windowLearner.act
		windowLearner
	}
	
	def learnBalconies(usedHogs:List[HOGDescriptor]) = {
		val balconyLearner = new Learner("balcony", List("door","window"), usedHogs)
		balconyLearner.act
		balconyLearner
	}
	
	def learnAll(usedHogs:List[HOGDescriptor]):Unit = {
		learnDoors(usedHogs)
		learnWindows(usedHogs)
		learnBalconies(usedHogs)
	}
}