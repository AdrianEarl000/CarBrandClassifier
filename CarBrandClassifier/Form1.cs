using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CarBrandClassifier
{
    public partial class Form1 : Form
    {
        private MLLoader MLLoader;

        public Form1()
        {
            InitializeComponent();
            MLLoader = new MLLoader(@"C:\Users\admin\source\repos\CarBrandClassifier\CarBrandClassifier\MLModel1.zip");

        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.ShowDialog();
            if (ofd.FileName != null)
            {
                pictureBox1.ImageLocation = ofd.FileName;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                // Convert the image to bytes
                ImageConverter converter = new ImageConverter();
                byte[] imageBytes = (byte[])converter.ConvertTo(pictureBox1.Image, typeof(byte[]));

                // Prepare input data for prediction
                ModelInput input = new ModelInput
                {
                    ImageSource = imageBytes
                };

                // Make prediction
                var prediction = MLLoader.Predict(input);

                // Display the predicted breed
                MessageBox.Show("Selected file is an image of a " + prediction.Prediction);
            }
            else
            {
                MessageBox.Show("Please select an image first.");
            }
        }
    }
}
