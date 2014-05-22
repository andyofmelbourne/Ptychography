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

# I think this is just for python files
#subprocess.Popen([sys.executable, "eg.py", "--pos=0.0"])

# send the job over ssh
#send = 'rsync -e ssh --recursive --progress --verbose --delete /home/amorgan/Physics/arkive/PETRA_Ptych/scan0181_exploration/cfelsgi_0/ amorgan@cfelsgi:/home/amorgan/andrew_morgan_temp/Ptychography/cfelsgi_0'

#try :
#    retcode = call(send, shell=True)
#    if retcode < 0:
#        print >>sys.stderr, "Child was terminated by signal", -retcode
#    else :
#        print >>sys.stderr, "Child returned", retcode
#except OSError as e:
#    print >>sys.stderr, "Execution failed:", e


#send = 'rsync -e ssh --recursive --progress --verbose --delete /home/amorgan/Physics/arkive/PETRA_Ptych/scan0181_exploration/cfelsgi_0/ amorgan@cfelsgi:/home/amorgan/andrew_morgan_temp/Ptychography/cfelsgi_0'
#try :
#    retcode = subprocess.call('ls', shell=True)
#    if retcode < 0:
#        print >>sys.stderr, "Child was terminated by signal", -retcode
#    else :
#        print >>sys.stderr, "Child returned", retcode
#except OSError as e:
#    print >>sys.stderr, "Execution failed:", e

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
        commands.append('python process_diffs.py -s0181 -r0 -o ../../../tempdata/MLL_calc/')
        #
        # Run the retrieval on the data 
        #commands.append('python ../python_scripts/Ptychography.py -i ../../../tempdata/MLL_calc/ -o ../../../tempdata/MLL_calc/')
        #
        # Process the results 
        #commands.append('python ../python_scripts/process_results.py -i ../../../tempdata/MLL_calc/ -o ../../../tempdata/MLL_calc/')
        #
        # execute sequentially
        for command in commands:
            subprocess.call(command, shell=True)
        #
        print 'I hope all went well!'
        print 'Your results should be in ../../../tempdata/MLL_calc/'


