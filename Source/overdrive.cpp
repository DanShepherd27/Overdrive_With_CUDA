#include <JuceHeader.h>
#include <string>
#include <iostream>
#include <juce_audio_formats/juce_audio_formats.h>

#include "./kernel.cuh"

class Overdrive {
public:
    Overdrive() {
        this->gain = 1;
    }
    Overdrive(int gain) {
        this->gain = gain;
    }

    int gain;

    void GPUApplyOverdrive(const float *const *samples_by_channels, int numOfChannels, int arrayLength, float originalMagnitude) {
        kernel(samples_by_channels, numOfChannels,  arrayLength, gain, originalMagnitude);
    }
};

//==============================================================================
int main ()
{

    // Reading file path

    std::string inputFilePath = "";

    std::cout << "Please enter your file path: ";

    std::cin >> inputFilePath;

    // Reading file

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    juce::File inputFile(inputFilePath);

    juce::AudioFormatReader* reader = formatManager.createReaderFor(inputFile);

    if (reader == nullptr) return 0;
    juce::AudioBuffer<float> buffer(2, (int)reader->lengthInSamples); // conversion from 'juce::int64' to 'int', possible loss of data


    bool successfulRead = reader->read(&buffer, 0, reader->lengthInSamples, 0, true, true);
    
    if (!successfulRead || buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
        std::cout << "The file read was unsuccessful.";
        return 0; 
    }

    std::cout << "Your file contains " << reader->lengthInSamples << " samples. \n";
    float originalMagnitude = buffer.getMagnitude(0, 0, buffer.getNumSamples());
    std::cout << "Magnitude before gain: " << originalMagnitude << std::endl;
    std::cout << "Please enter gain amount: ";
    int gain = 0;
    std::cin >> gain;

    Overdrive od = Overdrive(gain);
    od.GPUApplyOverdrive(buffer.getArrayOfReadPointers(), buffer.getNumChannels(), buffer.getNumSamples(), originalMagnitude);
    

    //std::cout << "Magnitude after gain: " << buffer.getMagnitude(0, 0, reader->lengthInSamples) << std::endl;
    
    //for (int sampleIndex = 0; sampleIndex < buffer.getNumSamples(); ++sampleIndex) {
    //    for (int channelIndex = 0; channelIndex < buffer.getNumChannels(); ++channelIndex) {
    //        float currentSample = buffer.getSample(channelIndex, sampleIndex);
    //        if (currentSample > originalMagnitude) buffer.setSample(channelIndex, sampleIndex, originalMagnitude);
    //        if (currentSample < -1 * originalMagnitude) buffer.setSample(channelIndex, sampleIndex, -1 * originalMagnitude);
    //    }
    //}

    //buffer.applyGain((float)(-1 * gain));

    // Save output to file
    std::cout << "Please specify the name of your exported file: ";
    std::string outputFilePath;
    std::cin >> outputFilePath;

    //juce::File outputFile = juce::File(juce::File::getCurrentWorkingDirectory().getChildFile(outputFilePath + inputFile.getFileExtension().toStdString()));
    juce::File outputFile = juce::File(juce::File::getCurrentWorkingDirectory().getChildFile(outputFilePath + ".wav"));
    //juce::AudioFormat* format = formatManager.findFormatForFileExtension(inputFile.getFileExtension());
    juce::FileOutputStream outputStream(outputFile);
    
    if (outputStream.failedToOpen()) return 0;


    outputStream.setPosition(0);
    outputStream.truncate();
    
    juce::AudioFormatWriter* formatWriter = juce::WavAudioFormat().createWriterFor(&outputStream, reader->sampleRate, buffer.getNumChannels(), reader->bitsPerSample, juce::StringPairArray(), 0);
    
    bool successfulWrite = formatWriter->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());

    if (!successfulWrite) return 0;
    
    std::cout << "Write successful." << std::endl;
    formatWriter->flush();
    delete reader;

    return 0;
}
