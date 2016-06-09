package com.exem;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/**
 * Created by khkim on 2016-06-09.
 */
public class CharacterIterator implements DataSetIterator {
    //Valid characters
    private char[] validCharacters;
    //Maps each character to an index ind the input/output
    private Map<Character,Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid
    private char[] fileCharacters;
    //Length of each example/minibatch (number of characters)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;
    private Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    public CharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, Random rng) throws IOException {
        if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        this.validCharacters = validCharacters;
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = new HashMap<>();
        for( int i=0; i<validCharacters.length; i++ ) charToIdxMap.put(validCharacters[i], i);

        //Load file and convert contents to a char[]
        boolean newLineValid = charToIdxMap.containsKey('\n');
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
        int maxSize = lines.size();	//add lines.size() to account for newline characters at end of each line
        for( String s : lines ) maxSize += s.length();
        char[] characters = new char[maxSize];
        int currIdx = 0;
        for( String s : lines ){
            char[] thisLine = s.toCharArray();
            for (char aThisLine : thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) continue;
                characters[currIdx++] = aThisLine;
            }
            if(newLineValid) characters[currIdx++] = '\n';
        }

        if( currIdx == characters.length ){
            fileCharacters = characters;
        } else {
            fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
        }
        if( exampleLength >= fileCharacters.length ) throw new IllegalArgumentException("exampleLength="+exampleLength
                +" cannot exceed number of valid characters in file ("+fileCharacters.length+")");

        int nRemoved = maxSize - fileCharacters.length;
        System.out.println("Loaded and converted file: " + fileCharacters.length + " valid characters of "
                + maxSize + " total characters (" + nRemoved + " removed)");

        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileCharacters.length-1) / exampleLength - 2;   //-2: for end index, and for partial example
        for( int i=0; i<nMinibatchesPerEpoch; i++ ){
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets,rng);
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    public static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for( char c : temp ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    /** As per getMinimalCharacterSet(), but with a few extra characters */
    public static char[] getDefaultCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c : getMinimalCharacterSet() ) validChars.add(c);
        char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
                '\\', '|', '<', '>'};
        for( char c : additionalChars ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    public char convertIndexToCharacter( int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.get(c);
    }

    public char getRandomCharacter(){
        return validCharacters[(int) (rng.nextDouble()*validCharacters.length)];
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = charToIdxMap.get(fileCharacters[j]);		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input,labels);
    }

    public int totalExamples() {
        return (fileCharacters.length-1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return validCharacters.length;
    }

    public int totalOutcomes() {
        return validCharacters.length;
    }

    public void reset() {
        exampleStartOffsets.clear();
        int nMinibatchesPerEpoch = totalExamples();
        for( int i=0; i<nMinibatchesPerEpoch; i++ ){
            exampleStartOffsets.add(i * miniBatchSize);
        }
        Collections.shuffle(exampleStartOffsets,rng);
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}

