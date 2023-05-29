#include <JuceHeader.h>
#include <string>
#include <iostream>
#include <juce_audio_formats/juce_audio_formats.h>
#include <string>

#include "./kernel.cuh"
#include "./PreciseTimer.h"

class Overdrive {
public:
    Overdrive() {
        this->gain = 1;
    }
    Overdrive(int gain) {
        this->gain = gain;
    }

    int gain;
    CPreciseTimer timer;

    void GPUApplyOverdrive(const float *const *samples_by_channels, int numOfChannels, int arrayLength, float originalMagnitude) {
        kernel(samples_by_channels, numOfChannels,  arrayLength, gain, originalMagnitude);
    }

    void CPUApplyOverdrive(juce::AudioBuffer<float> buffer) {
        float originalMagnitude = buffer.getMagnitude(0, 0, buffer.getNumSamples());

        timer.StartTimer();
        buffer.applyGain(gain);     

        for (int sampleIndex = 0; sampleIndex < buffer.getNumSamples(); ++sampleIndex) {
            for (int channelIndex = 0; channelIndex < buffer.getNumChannels(); ++channelIndex) {
                float currentSample = buffer.getSample(channelIndex, sampleIndex);
                if (currentSample > originalMagnitude) buffer.setSample(channelIndex, sampleIndex, originalMagnitude);
                if (currentSample < -1 * originalMagnitude) buffer.setSample(channelIndex, sampleIndex, -1 * originalMagnitude);
            }
        }
        timer.StopTimer();
        std::cout << "Timer: " << timer.GetTimeMilliSec() << "ms. \n";
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

    std::string mode_selector;
    do {
        std::cin.clear();
        std::cout << "Select computing mode: \n 1 - CPU \n 2 - GPU\n";
        std::cin >> mode_selector;
    } while (!(mode_selector == "1" || mode_selector == "2"));
    
    Overdrive od = Overdrive(gain);
    
    if (mode_selector == "1") {
        od.CPUApplyOverdrive(buffer);
    }
    else if(mode_selector == "2"){
        od.GPUApplyOverdrive(buffer.getArrayOfReadPointers(), buffer.getNumChannels(), buffer.getNumSamples(), originalMagnitude);
    }

    // Save output to file
    std::cout << "Please specify the name of your exported file: ";
    std::string outputFilePath;
    std::cin >> outputFilePath;

    juce::File outputFile = juce::File(juce::File::getCurrentWorkingDirectory().getChildFile(outputFilePath + ".wav"));
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
