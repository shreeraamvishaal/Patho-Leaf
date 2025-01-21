package com.example.patholeaftestcase

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.example.patholeaftestcase.databinding.FragmentSecondBinding
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.patholeaftestcase.ml.DiseaseDetection
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A simple [Fragment] subclass as the second destination in the navigation.
 */
class SecondFragment : Fragment() {

    private var _binding: FragmentSecondBinding? = null
    private val binding get() = _binding!!

    private val imageSize = 224 // default image size
    // This property is only valid between onCreateView and
    // onDestroyView.


    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        _binding = FragmentSecondBinding.inflate(inflater, container, false)

        binding.demoText.visibility = View.VISIBLE
        binding.clickHere.visibility = View.GONE
        binding.demoArrow.visibility = View.VISIBLE
        binding.classified.visibility = View.GONE
        binding.result.visibility = View.GONE
        binding.solution.visibility = View.GONE
        binding.solutionName.visibility = View.GONE

        binding.button.setOnClickListener {
            // Launch Camera if we have permission
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 1)
            } else {
                // request camera permission if don't have
                ActivityCompat.requestPermissions(requireActivity(), arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        return binding.root
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 1 && resultCode == AppCompatActivity.RESULT_OK) {
            val image = data?.extras?.get("data") as Bitmap
            val dimension = Math.min(image.width, image.height)
            var thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
            binding.imageView.setImageBitmap(thumbnail)

            binding.demoText.visibility = View.GONE
            binding.clickHere.visibility = View.VISIBLE
            binding.demoArrow.visibility = View.GONE
            binding.classified.visibility = View.VISIBLE
            binding.result.visibility = View.VISIBLE
            binding.solution.visibility = View.VISIBLE
            binding.solutionName.visibility = View.VISIBLE

            thumbnail = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
            classifyImage(thumbnail)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = DiseaseDetection.newInstance(requireContext())

            // Create input for reference
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            // Get 1D array of 224 * 224 pixels in image
            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)

            // Iterate over pixels and extract R, G, B values add to byteBuffer
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val value = intValues[pixel++] // RGB
                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat((value and 0xFF) * (1f / 255f))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            // Run model inference and get result
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidence = outputFeature0.floatArray

            // Find the index of the class with the highest confidence
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidence.indices) {
                if (confidence[i] > maxConfidence) {
                    maxConfidence = confidence[i]
                    maxPos = i
                }
            }
            val classes = arrayOf("Pepper Bell Bacterial Spot", "Potato Early Blight", "Potato Late Blight", "Tomato Bacterial Spot", "Tomato Tomato YellowLeaf Curl Virus")
            binding.result.text = classes[maxPos]
            binding.result.setOnClickListener {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + binding.result.text)))
            }

            model.close()

        } catch (e: IOException) {
            // TODO Handle the exception
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
//    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
//        super.onViewCreated(view, savedInstanceState)
//
////        binding.buttonSecond.setOnClickListener {
////            findNavController().navigate(R.id.action_SecondFragment_to_FirstFragment)
////        }
//    }


//}