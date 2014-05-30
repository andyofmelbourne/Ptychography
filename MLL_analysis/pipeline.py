# This script controls the basic workflow for a Ptychographic retrieval from cfelsgi:
#                 
# send ~/Physics/git_repos/Ptychoraphy -->  cfelsgi/home/amorgan/analysis                
#                 
# remotely execute process_diffs.py: (../../../rawdata) .h5  --> (../../../tempdata/MLL_calc) diffs + inital variables 
#                 
# remotely execute script: MLL_ptych.py --> results (~/Physics/tempdata/MLL_calc)
#
# (cfelgi/home/amorgan/tempdata/MLL_calc) --> (~/Physics/tempdata/MLL_calc)
#
# locally run analysis script process_results.py --> tempdata

import sys
import os, sys, getopt
import subprocess 

# Parameters
##########################################
##########################################
tempdata_dir = '../../../tempdata/MLL_calc/'

sequence = """
pmod_int = True
Thibault_sample = 10
Thibault_both = 50
"""

gratingSim = False
samplesupport = True
sample1d = True
##########################################
##########################################

def main(argv):
    location = 'local'
    try :
        opts, args = getopt.getopt(argv,"hl:",["location="])
    except getopt.GetoptError:
        print 'python pipeline -l <location>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python pipeline -l <local/cfelsgi>'
            sys.exit()
        elif opt in ("-l", "--location"):
            location = arg
    return location

if __name__=='__main__':
    loc = main(sys.argv[1:])
    if loc == 'local':
        commands = []
        #
        # process the .h5 files
        commands.append('python process_diffs.py --scan=0181 --run=0 --outputdir=' + tempdata_dir + ' --samplesupport='+str(samplesupport)+' --sample1d='+str(sample1d)+' --gratingSim='+str(gratingSim))
        #
        # Output the algorithm parameters
        print '#########################################################'
        print 'Outputing the sequence.txt file...'
        with open(tempdata_dir + "sequence.txt", "w") as text_file:
            text_file.write(sequence)
        print '#########################################################'
        #
        # Run the retrieval on the data 
        commands.append('python ../python_scripts/Ptychography.py -i ' +tempdata_dir+ ' -o ' +tempdata_dir)
        #
        # Process the results 
        commands.append('python process_results.py -i ' +tempdata_dir+ ' -o ' +tempdata_dir)
        #
        # execute sequentially
        for command in commands:
            print command
            try :
                retcode = subprocess.call(command, shell=True)
                if retcode < 0:
                    print >>sys.stderr, "Child was terminated by signal", -retcode
            except OSError as e:
                print >>sys.stderr, "Execution failed:", e
        #
        print 'I hope all went well!'
        print 'Your results should be in ' + tempdata_dir 
        #
    elif loc == 'cfelsgi':
        # send ../Ptychography to cfelsgi 
        # then run this script "locally" on cfelsgi by remotely sending the command over ssh
        # then call in tempdata from cfelsgi to here
        # This only works if the relative directory trees are the same
        commands = []
        #
        # send the job over ssh
        commands.append('rsync -qe ssh --recursive --progress --delete ../../Ptychography/ amorgan@cfelsgi:/home/amorgan/analysis/Ptychography')
        #
        # Run this script
        commands.append("ssh cfelsgi 'export LD_LIBRARY_PATH='/cfel/common/lib:$LD_LIBRARY_PATH'; export PYTHONPATH='/nfs/cfel/cxi/common/cfelsgi/gcc_4_4_7/python-hdf5/2.3.0/lib64/python2.6/site-packages:$PYTHONPATH'; cd /home/amorgan/analysis/Ptychography/MLL_analysis/; python pipeline.py -l local'")
        #
        # retrieve the data
        commands.append('rsync -qe ssh --recursive --progress amorgan@cfelsgi:/home/amorgan/tempdata/ ../../../tempdata/')
        #
        # execute sequentially
        for command in commands:
            print command
            try :
                retcode = subprocess.call(command, shell=True)
                if retcode < 0:
                    print >>sys.stderr, "Child was terminated by signal", -retcode
            except OSError as e:
                print >>sys.stderr, "Execution failed:", e
        # 


